# MIDI Master Troubleshooting Guide

This document covers common issues you might encounter when using MIDI Master and their solutions.

## Installation Issues

### ImportError: No module named 'mido'

**Problem**: You receive an error about the mido library not being found.

**Solution**: Install the required dependency:
```bash
pip install mido
```

Or install all dependencies:
```bash
pip install -r requirements.txt
```

### Python Version Error

**Problem**: The program fails to run with syntax errors.

**Solution**: Ensure you're using Python 3.6 or higher:
```bash
python --version
```

If you have an older version, install Python 3.6+ from [python.org](https://www.python.org/downloads/).

## Runtime Issues

### Generated MIDI file sounds strange

**Problem**: The output MIDI file doesn't sound like proper music.

**Solutions**:
1. Check that your DAW's virtual instruments are properly assigned
2. Try different genres and moods for variety
3. Adjust tempo to better match the genre
4. Verify that the MIDI file plays correctly in a standard MIDI player

### Program crashes with error messages

**Problem**: The program exits with error messages.

**Solutions**:
1. Ensure you're using Python 3.6 or higher
2. Check that all dependencies are installed
3. Verify command-line arguments are valid
4. Check that you have write permissions in the output directory

### No output file is created

**Problem**: The program runs without errors but no MIDI file is created.

**Solutions**:
1. Check that you have write permissions in the directory
2. Verify the output file path is valid
3. Ensure the program completes without errors
4. Check that the output directory exists

## Genre-Specific Issues

### Pop Music

**Problem**: Pop songs sound too repetitive.

**Solution**: Try different moods or increase the number of bars for more variation.

### Rock Music

**Problem**: Rock songs lack energy.

**Solution**: Use the 'energetic' mood and higher tempo values (120-160 BPM).

### Jazz Music

**Problem**: Jazz songs sound too simple.

**Solution**: Jazz is complex by nature. Try increasing the number of bars or using the 'energetic' mood for more complex progressions.

### Electronic Music

**Problem**: Electronic tracks don't sound danceable.

**Solution**: Use tempo values around 120-130 BPM which is typical for dance music.

### Hip-Hop Music

**Problem**: Hip-hop tracks lack groove.

**Solution**: Use tempo values around 80-90 BPM which is typical for hip-hop.

### Classical Music

**Problem**: Classical pieces sound too modern.

**Solution**: Use slower tempos (70-100 BPM) and the 'calm' or 'sad' moods for more traditional classical sounds.

## MIDI Output Issues

### File won't import into DAW

**Problem**: Your DAW reports the MIDI file is corrupted or invalid.

**Solutions**:
1. Verify the program completed without errors
2. Try playing the file in a standard MIDI player first
3. Check that the output file path is valid and writable
4. Ensure you're using a recent version of your DAW

### Wrong instruments in DAW

**Problem**: The MIDI file plays with unexpected instruments.

**Solution**: Assign appropriate virtual instruments to different MIDI tracks in your DAW. MIDI Master doesn't specify instruments, so your DAW uses default assignments.

### Timing issues

**Problem**: Notes seem to play at the wrong time or with incorrect durations.

**Solutions**:
1. Check your DAW's tempo settings match the MIDI file
2. Verify your DAW's time signature settings (MIDI Master uses 4/4)
3. Try a different DAW or MIDI player to isolate the issue

## Development Issues

### Import errors in example scripts

**Problem**: Example scripts fail with import errors.

**Solution**: Ensure you're running the scripts from the correct directory (the project root) so that the imports can find the modules.

### Adding new genres

**Problem**: Added a new genre but it's not recognized.

**Solutions**:
1. Verify the genre name is correctly spelled and lowercase
2. Ensure the new genre class is added to the genre_map in GenreFactory
3. Check that the new genre class inherits from GenreRules
4. Verify all required methods are implemented

## Performance Issues

### Slow generation times

**Problem**: Generating songs takes a long time.

**Solutions**:
1. Reduce the number of bars being generated
2. Ensure you're using a recent version of Python
3. Close other applications to free up system resources

### Memory errors

**Problem**: The program crashes with memory errors.

**Solutions**:
1. Reduce the number of bars being generated
2. Generate shorter songs (fewer bars)
3. Close other applications to free up system memory

## Getting Help

If you encounter issues not covered here:

1. Check the console output for detailed error messages
2. Verify all installation steps were completed correctly
3. Ensure command-line arguments are properly formatted
4. Check that your Python environment is properly configured
5. Search online for the specific error message
6. Consider filing an issue on the project's GitHub repository (if available)

## Contact

For additional support, please:
1. Provide the exact error message you're seeing
2. Include the command you ran
3. Specify your operating system and Python version
4. Describe what you were trying to accomplish