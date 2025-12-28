#include "RLBotBridge.h"
#include <sstream>
#include <cstring>

BAKKESMOD_PLUGIN(RLBotBridge, "RLBot Bridge", "1.0", PLUGINTYPE_FREEPLAY)

void RLBotBridge::onLoad()
{
    isRunning = true;
    isConnected = false;
    botEnabled = false;
    latestActions.resize(ACTION_SIZE, 0.0f);
    
    cvarManager->log("RLBot Bridge plugin loaded");
    
    // Register physics tick hook
    gameWrapper->HookEvent("Function TAGame.Car_TA.SetVehicleInput", 
        std::bind(&RLBotBridge::OnPhysicsTick, this, std::placeholders::_1));
    
    // Register F1 hotkey for toggling bot
    cvarManager->registerNotifier("rlbot_toggle", 
        [this](std::vector<std::string> args) { ToggleBotControl(); }, 
        "Toggle bot control on/off", 
        PERMISSION_ALL);
    
    // Initialize socket server
    InitializeSocket();
    
    cvarManager->log("RLBot Bridge: Press F1 to toggle bot control");
}

void RLBotBridge::onUnload()
{
    isRunning = false;
    botEnabled = false;
    
    // Wait for socket thread to finish
    if (socketThread.joinable()) {
        socketThread.join();
    }
    
    CleanupSocket();
    cvarManager->log("RLBot Bridge plugin unloaded");
}

void RLBotBridge::InitializeSocket()
{
    WSADATA wsaData;
    int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (result != 0) {
        cvarManager->log("WSAStartup failed: " + std::to_string(result));
        return;
    }
    
    serverSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (serverSocket == INVALID_SOCKET) {
        cvarManager->log("Socket creation failed");
        WSACleanup();
        return;
    }
    
    // Set socket to non-blocking mode
    u_long mode = 1;
    ioctlsocket(serverSocket, FIONBIO, &mode);
    
    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_addr.s_addr = INADDR_ANY;
    serverAddr.sin_port = htons(PORT);
    
    if (bind(serverSocket, (sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
        cvarManager->log("Bind failed: " + std::to_string(WSAGetLastError()));
        closesocket(serverSocket);
        WSACleanup();
        return;
    }
    
    if (listen(serverSocket, 1) == SOCKET_ERROR) {
        cvarManager->log("Listen failed");
        closesocket(serverSocket);
        WSACleanup();
        return;
    }
    
    cvarManager->log("RLBot Bridge: Socket server listening on port " + std::to_string(PORT));
    
    // Start socket server thread
    socketThread = std::thread(&RLBotBridge::SocketServerThread, this);
}

void RLBotBridge::SocketServerThread()
{
    clientSocket = INVALID_SOCKET;
    
    while (isRunning) {
        if (!isConnected) {
            // Try to accept connection
            sockaddr_in clientAddr;
            int clientAddrSize = sizeof(clientAddr);
            SOCKET newClient = accept(serverSocket, (sockaddr*)&clientAddr, &clientAddrSize);
            
            if (newClient != INVALID_SOCKET) {
                clientSocket = newClient;
                isConnected = true;
                cvarManager->log("RLBot Bridge: Client connected");
                
                // Set client socket to blocking mode
                u_long mode = 0;
                ioctlsocket(clientSocket, FIONBIO, &mode);
            }
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void RLBotBridge::CleanupSocket()
{
    if (clientSocket != INVALID_SOCKET) {
        closesocket(clientSocket);
        clientSocket = INVALID_SOCKET;
    }
    
    if (serverSocket != INVALID_SOCKET) {
        closesocket(serverSocket);
        serverSocket = INVALID_SOCKET;
    }
    
    WSACleanup();
}

void RLBotBridge::ToggleBotControl()
{
    botEnabled = !botEnabled;
    if (botEnabled) {
        cvarManager->log("RLBot Bridge: Bot control ENABLED");
    } else {
        cvarManager->log("RLBot Bridge: Bot control DISABLED");
    }
}

void RLBotBridge::OnPhysicsTick(std::string eventName)
{
    if (!botEnabled || !isConnected) {
        return;
    }
    
    ServerWrapper server = gameWrapper->GetCurrentGameState();
    if (!server) return;
    
    // Extract observations
    std::vector<float> observations = ExtractObservations();
    
    if (observations.empty()) {
        return;
    }
    
    // Send observations to Python client
    std::lock_guard<std::mutex> lock(socketMutex);
    
    try {
        // Send observation size first (4 bytes)
        int obsSize = observations.size();
        send(clientSocket, (char*)&obsSize, sizeof(int), 0);
        
        // Send observations (float array)
        int bytesSent = send(clientSocket, (char*)observations.data(), 
                            observations.size() * sizeof(float), 0);
        
        if (bytesSent == SOCKET_ERROR) {
            cvarManager->log("Send failed, disconnecting");
            isConnected = false;
            closesocket(clientSocket);
            clientSocket = INVALID_SOCKET;
            return;
        }
        
        // Receive actions from Python
        std::vector<float> actions(ACTION_SIZE);
        int bytesReceived = recv(clientSocket, (char*)actions.data(), 
                                ACTION_SIZE * sizeof(float), 0);
        
        if (bytesReceived == SOCKET_ERROR || bytesReceived == 0) {
            cvarManager->log("Receive failed, disconnecting");
            isConnected = false;
            closesocket(clientSocket);
            clientSocket = INVALID_SOCKET;
            return;
        }
        
        // Apply actions to the car
        ApplyActions(actions);
        
    } catch (const std::exception& e) {
        cvarManager->log("Exception in physics tick: " + std::string(e.what()));
        isConnected = false;
    } catch (...) {
        cvarManager->log("Unknown exception in physics tick");
        isConnected = false;
    }
}

std::vector<float> RLBotBridge::ExtractObservations()
{
    std::vector<float> obs;
    obs.reserve(OBSERVATION_SIZE);
    
    ServerWrapper server = gameWrapper->GetCurrentGameState();
    if (!server) return obs;
    
    BallWrapper ball = server.GetBall();
    if (!ball) return obs;
    
    CarWrapper car = gameWrapper->GetLocalCar();
    if (!car) return obs;
    
    // Ball data (18 values)
    Vector ballLoc = ball.GetLocation();
    Vector ballVel = ball.GetVelocity();
    Rotator ballRot = ball.GetRotation();
    Vector ballAngVel = ball.GetAngularVelocity();
    
    // Normalize positions (divide by field dimensions)
    obs.push_back(ballLoc.X / 4096.0f);
    obs.push_back(ballLoc.Y / 5120.0f);
    obs.push_back(ballLoc.Z / 2044.0f);
    
    // Normalize velocities (divide by max car speed ~2300)
    obs.push_back(ballVel.X / 2300.0f);
    obs.push_back(ballVel.Y / 2300.0f);
    obs.push_back(ballVel.Z / 2300.0f);
    
    // Ball rotation (normalized)
    obs.push_back(ballRot.Pitch / 32768.0f);
    obs.push_back(ballRot.Yaw / 32768.0f);
    obs.push_back(ballRot.Roll / 32768.0f);
    
    // Ball angular velocity (normalized)
    obs.push_back(ballAngVel.X / 5.5f);
    obs.push_back(ballAngVel.Y / 5.5f);
    obs.push_back(ballAngVel.Z / 5.5f);
    
    // Player car data (46 values per car, we'll do for the local player)
    Vector carLoc = car.GetLocation();
    Vector carVel = car.GetVelocity();
    Rotator carRot = car.GetRotation();
    Vector carAngVel = car.GetAngularVelocity();
    
    // Car position (normalized)
    obs.push_back(carLoc.X / 4096.0f);
    obs.push_back(carLoc.Y / 5120.0f);
    obs.push_back(carLoc.Z / 2044.0f);
    
    // Car velocity (normalized)
    obs.push_back(carVel.X / 2300.0f);
    obs.push_back(carVel.Y / 2300.0f);
    obs.push_back(carVel.Z / 2300.0f);
    
    // Car rotation (normalized)
    obs.push_back(carRot.Pitch / 32768.0f);
    obs.push_back(carRot.Yaw / 32768.0f);
    obs.push_back(carRot.Roll / 32768.0f);
    
    // Car angular velocity (normalized)
    obs.push_back(carAngVel.X / 5.5f);
    obs.push_back(carAngVel.Y / 5.5f);
    obs.push_back(carAngVel.Z / 5.5f);
    
    // Car forward, up, right vectors (rotation matrix)
    Vector forward = car.GetForwardVector();
    Vector up = car.GetUpVector();
    Vector right = car.GetRightVector();
    
    obs.push_back(forward.X);
    obs.push_back(forward.Y);
    obs.push_back(forward.Z);
    obs.push_back(up.X);
    obs.push_back(up.Y);
    obs.push_back(up.Z);
    obs.push_back(right.X);
    obs.push_back(right.Y);
    obs.push_back(right.Z);
    
    // Boost amount
    BoostWrapper boostWrapper = car.GetBoostComponent();
    float boostAmount = boostWrapper ? boostWrapper.GetCurrentBoostAmount() / 100.0f : 0.0f;
    obs.push_back(boostAmount);
    
    // Car states (on ground, etc)
    obs.push_back(car.IsOnGround() ? 1.0f : 0.0f);
    obs.push_back(car.IsOnWall() ? 1.0f : 0.0f);
    obs.push_back(car.HasFlip() ? 1.0f : 0.0f);
    obs.push_back(car.IsJumping() ? 1.0f : 0.0f);
    
    // Fill remaining observations with zeros to reach 159
    // In a full implementation, this would include:
    // - Other cars data (5 more cars * 46 values = 230)
    // - Boost pads (34 pads * 2 values = 68)
    // - Game state info
    // For now, we'll pad to 159
    while (obs.size() < OBSERVATION_SIZE) {
        obs.push_back(0.0f);
    }
    
    return obs;
}

void RLBotBridge::ApplyActions(const std::vector<float>& actions)
{
    if (actions.size() < ACTION_SIZE) {
        return;
    }
    
    CarWrapper car = gameWrapper->GetLocalCar();
    if (!car) return;
    
    ControllerInput input;
    
    // Actions: [Throttle, Steer, Pitch, Yaw, Roll, Jump, Boost, Handbrake]
    input.Throttle = actions[0];    // -1 to 1
    input.Steer = actions[1];       // -1 to 1
    input.Pitch = actions[2];       // -1 to 1
    input.Yaw = actions[3];         // -1 to 1
    input.Roll = actions[4];        // -1 to 1
    input.Jump = actions[5] > 0.5f; // Boolean
    input.Boost = actions[6] > 0.5f; // Boolean
    input.Handbrake = actions[7] > 0.5f; // Boolean
    
    car.SetInput(input);
}
