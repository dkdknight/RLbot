# Contributing to RLBot Bridge

Thank you for your interest in contributing to RLBot Bridge! This guide will help you get started.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, etc.)
- Error messages or logs

### Suggesting Features

Feature requests are welcome! Please include:
- Clear description of the feature
- Use case and motivation
- Example implementation (if applicable)

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Test thoroughly**:
   - Run existing tests: `python test_bot_manager.py`
   - Test with actual game if possible
5. **Commit with clear messages**
6. **Push to your fork**
7. **Open a Pull Request**

## Development Setup

### Python Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
python test_bot_manager.py
```

### C++ Plugin Development

1. Install Visual Studio 2019+ with C++ tools
2. Download BakkesMod SDK
3. Update `CMakeLists.txt` with SDK path
4. Build:
   ```bash
   cd BakkesModPlugin
   mkdir build
   cd build
   cmake .. -G "Visual Studio 16 2019" -A x64
   cmake --build . --config Release
   ```

## Code Style

### Python
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to functions/classes
- Keep functions focused and modular

### C++
- Follow BakkesMod plugin conventions
- Use meaningful variable names
- Comment complex logic
- Handle errors gracefully

## Testing

### Python Tests
- Add tests for new features in `test_bot_manager.py`
- Ensure all existing tests still pass
- Test with different model types

### Plugin Testing
- Test in Freeplay mode first
- Verify socket communication
- Check for memory leaks
- Test F1 toggle functionality

## Documentation

When adding features:
- Update README.md if architecture changes
- Update INSTALL.md if setup process changes
- Add comments to complex code
- Update relevant documentation files

## Project Structure

```
RLbot/
├── BakkesModPlugin/      # C++ plugin code
├── bots/                 # Bot models directory
├── main.py              # Python bot manager
├── test_bot_manager.py  # Test suite
└── *.md                 # Documentation
```

## Commit Messages

Use clear, descriptive commit messages:
```
Good:
- "Add LSTM hidden state reset on disconnect"
- "Fix frame skip counter overflow"
- "Update README with CUDA setup instructions"

Bad:
- "fix bug"
- "update"
- "changes"
```

## Pull Request Guidelines

Your PR should:
- Have a clear title and description
- Reference related issues
- Include tests for new features
- Update documentation as needed
- Pass all existing tests
- Be focused on a single feature/fix

## Areas for Contribution

### High Priority
- Multi-agent support (multiple bots)
- Enhanced observation space (all boost pads)
- Performance optimizations
- Better error handling

### Medium Priority
- GUI for bot selection
- Model hot-reloading
- Recording/replay system
- Cross-platform support

### Documentation
- Video tutorials
- More example bots
- Troubleshooting guides
- Translation to other languages

## Getting Help

- Check existing documentation
- Look at closed issues for similar problems
- Ask questions in GitHub Discussions
- Be patient and respectful

## Code of Conduct

Be respectful, constructive, and collaborative:
- Be welcoming to newcomers
- Provide constructive feedback
- Focus on what is best for the project
- Show empathy towards others

## License

By contributing, you agree that your contributions will be licensed under the same terms as the project (see LICENSE file).

## Recognition

Contributors will be recognized in:
- GitHub contributors list
- Release notes for significant contributions
- README acknowledgments section

## Thank You!

Every contribution helps make RLBot Bridge better. Whether it's code, documentation, bug reports, or feature ideas - thank you for contributing! 🚀
