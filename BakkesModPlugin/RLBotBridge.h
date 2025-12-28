#pragma once

#include "bakkesmod/plugin/bakkesmodplugin.h"
#include "bakkesmod/plugin/pluginwindow.h"
#include <winsock2.h>
#include <ws2tcpip.h>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>

#pragma comment(lib, "Ws2_32.lib")

class RLBotBridge : public BakkesMod::Plugin::BakkesModPlugin
{
private:
    // Socket communication
    SOCKET serverSocket;
    SOCKET clientSocket;
    std::thread socketThread;
    std::atomic<bool> isRunning;
    std::atomic<bool> isConnected;
    std::mutex socketMutex;
    
    // Bot control
    std::atomic<bool> botEnabled;
    std::vector<float> latestActions;
    std::mutex actionMutex;
    
    // Constants
    const int PORT = 5000;
    const int OBSERVATION_SIZE = 159;
    const int ACTION_SIZE = 8;
    
    // Methods
    void InitializeSocket();
    void SocketServerThread();
    void CleanupSocket();
    
    void OnPhysicsTick(std::string eventName);
    void ToggleBotControl();
    
    std::vector<float> ExtractObservations();
    void ApplyActions(const std::vector<float>& actions);
    
    Vector GetBallLocation();
    Vector GetBallVelocity();
    Rotator GetBallRotation();
    Vector GetBallAngularVelocity();

public:
    virtual void onLoad();
    virtual void onUnload();
};
