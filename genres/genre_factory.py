"""
Genre factory for creating genre-specific rules in the MIDI Master music generation program.

This module implements the Factory design pattern to create genre-specific rule objects.
The factory abstracts the creation process, allowing the rest of the program to work
with genre rules without knowing the specific implementation classes.

The factory maps genre names to their corresponding rule classes and provides
a simple interface for creating genre rule objects.
"""

from typing import Dict, Any
from .genre_rules import (
    PopRules, RockRules, JazzRules,
    ElectronicRules, HipHopRules, ClassicalRules, DnBRules, GenreRules
)


class GenreFactory:
    """Factory for creating genre-specific rules.
    
    This factory class implements the Factory design pattern to create
    genre-specific rule objects. It maps genre names to their corresponding
    rule classes and provides a simple interface for instantiation.
    
    The factory ensures that:
    1. Only valid genres can be created
    2. Genre creation is centralized in one place
    3. The rest of the program doesn't need to know about specific rule classes
    """
    
    @staticmethod
    def create_genre_rules(genre: str) -> GenreRules:
        """
        Create genre-specific rules based on the genre.
        
        This method takes a genre name and returns an instance of the appropriate
        GenreRules subclass for that genre.
        
        Args:
            genre: The genre to create rules for (e.g., 'pop', 'rock', 'jazz')
            
        Returns:
            An instance of a GenreRules subclass.
            
        Raises:
            ValueError: If the genre is not supported
        """
        # Map genre names to their corresponding rule classes
        genre_map = {
            'pop': PopRules,
            'rock': RockRules,
            'jazz': JazzRules,
            'electronic': ElectronicRules,
            'hip-hop': HipHopRules,
            'classical': ClassicalRules,
            'dnb': DnBRules,
            'drum-and-bass': DnBRules
        }
        
        # Validate that the requested genre is supported
        if genre not in genre_map:
            raise ValueError(f"Unsupported genre: {genre}")
            
        # Create an instance of the appropriate rule class
        return genre_map[genre]()