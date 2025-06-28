//! Generic lexer framework for tokenizing input streams
//!
//! This module provides the core tokenization infrastructure that can be
//! specialized for different RDF formats.

use crate::error::{TextPosition, TokenRecognizerError};
use std::io::{BufRead, Read};

/// A token recognizer that converts byte streams into tokens
pub trait TokenRecognizer {
    /// The token type produced by this recognizer
    type Token<'a>;
    
    /// Configuration options for tokenization
    type Options: Default;
    
    /// Recognize the next token from input data
    ///
    /// Returns:
    /// - `Some((consumed_bytes, Ok(token)))` for successful recognition
    /// - `Some((consumed_bytes, Err(error)))` for tokenization errors
    /// - `None` if more data is needed
    fn recognize_next_token<'a>(
        &mut self,
        data: &'a [u8],
        is_ending: bool,
        options: &Self::Options,
    ) -> Option<(usize, Result<Self::Token<'a>, TokenRecognizerError>)>;
    
    /// Reset the recognizer state
    fn reset(&mut self) {}
}

/// A token or a line jump in the input
#[derive(Debug, Clone, PartialEq)]
pub enum TokenOrLineJump<T> {
    /// A regular token
    Token(T),
    /// A line jump (newline character)
    LineJump,
}

/// A streaming tokenizer that processes input incrementally
pub struct StreamingTokenizer<R, T: TokenRecognizer> {
    recognizer: T,
    reader: R,
    buffer: Vec<u8>,
    position: TextPosition,
    options: T::Options,
    buffer_capacity: usize,
}

impl<R: BufRead, T: TokenRecognizer> StreamingTokenizer<R, T> {
    /// Create a new streaming tokenizer
    pub fn new(recognizer: T, reader: R) -> Self {
        Self {
            recognizer,
            reader,
            buffer: Vec::new(),
            position: TextPosition::start(),
            options: T::Options::default(),
            buffer_capacity: 8192, // 8KB default buffer
        }
    }
    
    /// Create a new streaming tokenizer with custom options
    pub fn with_options(recognizer: T, reader: R, options: T::Options) -> Self {
        Self {
            recognizer,
            reader,
            buffer: Vec::new(),
            position: TextPosition::start(),
            options,
            buffer_capacity: 8192,
        }
    }
    
    /// Set the buffer capacity
    pub fn with_buffer_capacity(mut self, capacity: usize) -> Self {
        self.buffer_capacity = capacity;
        self
    }
    
    /// Get the current position in the input
    pub fn position(&self) -> TextPosition {
        self.position
    }
}

impl<R: BufRead, T: TokenRecognizer> Iterator for StreamingTokenizer<R, T> {
    type Item = Result<TokenOrLineJump<T::Token<'static>>, TokenRecognizerError>;
    
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Clone buffer data to avoid borrowing issues
            let buffer_data = self.buffer.clone();
            
            // Try to recognize a token from current buffer
            let recognition_result = self.recognizer.recognize_next_token(
                &buffer_data,
                false, // We'll handle EOF separately
                &self.options,
            );
            
            if let Some((consumed, result)) = recognition_result {
                // Update position and consume bytes
                self.position.advance_bytes(&self.buffer[..consumed]);
                self.buffer.drain(..consumed);
                
                return Some(result.map(|token| {
                    // Convert token to owned version
                    // This is a simplification - in practice we'd need trait bounds
                    // for converting borrowed tokens to owned tokens
                    unsafe { std::mem::transmute(TokenOrLineJump::Token(token)) }
                }));
            }
            
            // Need more data - read from input
            let mut chunk = vec![0u8; self.buffer_capacity];
            match self.reader.read(&mut chunk) {
                Ok(0) => {
                    // EOF - try to recognize with ending flag
                    if !self.buffer.is_empty() {
                        let buffer_data = self.buffer.clone();
                        let eof_recognition_result = self.recognizer.recognize_next_token(
                            &buffer_data,
                            true,
                            &self.options,
                        );
                        
                        if let Some((consumed, result)) = eof_recognition_result {
                            self.position.advance_bytes(&self.buffer[..consumed]);
                            self.buffer.drain(..consumed);
                            return Some(result.map(|token| {
                                unsafe { std::mem::transmute(TokenOrLineJump::Token(token)) }
                            }));
                        }
                    }
                    return None; // True EOF
                }
                Ok(n) => {
                    chunk.truncate(n);
                    self.buffer.extend_from_slice(&chunk);
                }
                Err(e) => {
                    return Some(Err(TokenRecognizerError::UnexpectedCharacter(
                        e.to_string().chars().next().unwrap_or('?')
                    )));
                }
            }
        }
    }
}

/// A simple tokenizer for line-based formats (N-Triples, N-Quads)
pub struct LineTokenizer<R> {
    reader: R,
    position: TextPosition,
}

impl<R: BufRead> LineTokenizer<R> {
    /// Create a new line tokenizer
    pub fn new(reader: R) -> Self {
        Self {
            reader,
            position: TextPosition::start(),
        }
    }
    
    /// Get the current position
    pub fn position(&self) -> TextPosition {
        self.position
    }
}

impl<R: BufRead> Iterator for LineTokenizer<R> {
    type Item = Result<String, std::io::Error>;
    
    fn next(&mut self) -> Option<Self::Item> {
        let mut line = String::new();
        match self.reader.read_line(&mut line) {
            Ok(0) => None, // EOF
            Ok(_) => {
                // Update position
                for ch in line.chars() {
                    self.position.advance_char(ch);
                }
                
                // Remove trailing newline
                if line.ends_with('\n') {
                    line.pop();
                    if line.ends_with('\r') {
                        line.pop();
                    }
                }
                
                Some(Ok(line))
            }
            Err(e) => Some(Err(e)),
        }
    }
}