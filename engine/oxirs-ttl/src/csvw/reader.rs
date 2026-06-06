//! RFC 4180 CSV reader — pure Rust, no external `csv` crate.
//!
//! Implements just enough of RFC 4180 to handle real-world CSVW data files:
//! - Comma (or custom) delimiter
//! - Double-quote quoting with `""` escape for embedded quotes
//! - Quoted fields may span multiple lines (embedded CRLF/LF)
//! - CRLF and bare LF both accepted as record terminators
//! - Empty fields (consecutive delimiters) produce empty strings

use std::io::Read;

use super::CsvwError;

// ────────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────────

/// A single parsed CSV record (one data row).
#[derive(Debug, Clone, PartialEq)]
pub struct CsvRecord {
    /// The individual field values in left-to-right order.
    pub fields: Vec<String>,
}

/// RFC 4180 CSV reader built on top of any [`Read`] source.
///
/// The reader buffers the entire source into memory on first use so that
/// quoted newlines and CR+LF line endings are handled correctly without
/// the complexity of a byte-level state machine operating on partial reads.
pub struct CsvReader<R: Read> {
    inner: R,
    delimiter: u8,
    quote: u8,
    buf: Vec<u8>,
    /// Byte position within `buf` for the current parse pass.
    pos: usize,
    /// 1-based line counter (for error messages).
    line: usize,
    /// Whether the source has been fully loaded into `buf`.
    loaded: bool,
}

impl<R: Read> CsvReader<R> {
    /// Create a new reader with the standard comma delimiter and double-quote quoting.
    pub fn new(inner: R) -> Self {
        Self::with_delimiter(inner, b',')
    }

    /// Create a new reader with a custom delimiter byte.
    pub fn with_delimiter(inner: R, delimiter: u8) -> Self {
        Self {
            inner,
            delimiter,
            quote: b'"',
            buf: Vec::new(),
            pos: 0,
            line: 1,
            loaded: false,
        }
    }

    // ── private helpers ─────────────────────────────────────────────────────

    /// Ensure the full source has been read into `self.buf`.
    fn ensure_loaded(&mut self) -> Result<(), CsvwError> {
        if !self.loaded {
            self.inner
                .read_to_end(&mut self.buf)
                .map_err(|e| CsvwError::IoError(e.to_string()))?;
            self.loaded = true;
        }
        Ok(())
    }

    /// Peek at the byte at `self.pos` without advancing.
    fn peek(&self) -> Option<u8> {
        self.buf.get(self.pos).copied()
    }

    /// Consume and return the byte at `self.pos`.
    fn next_byte(&mut self) -> Option<u8> {
        let byte = self.buf.get(self.pos).copied();
        if byte.is_some() {
            self.pos += 1;
        }
        byte
    }

    /// After consuming a CR, skip the following LF (if any) and advance the
    /// line counter.
    fn skip_cr_if_lf(&mut self) {
        if self.peek() == Some(b'\n') {
            self.pos += 1;
        }
        // Either CRLF or lone CR — both end a line.
        self.line += 1;
    }

    /// Parse a single quoted field.  We have already consumed the opening
    /// quote byte before entering this function.
    fn parse_quoted_field(&mut self) -> Result<String, CsvwError> {
        let mut value = Vec::new();
        let start_line = self.line;

        loop {
            match self.next_byte() {
                None => {
                    return Err(CsvwError::CsvError {
                        line: start_line,
                        msg: "unterminated quoted field".into(),
                    });
                }
                Some(byte) if byte == self.quote => {
                    // Either `""` (escaped quote) or end of field.
                    if self.peek() == Some(self.quote) {
                        // Escaped double-quote: consume second and push one.
                        self.pos += 1;
                        value.push(self.quote);
                    } else {
                        // End of quoted field.
                        break;
                    }
                }
                Some(b'\r') => {
                    // CRLF inside quoted field — normalise to LF.
                    if self.peek() == Some(b'\n') {
                        self.pos += 1;
                    }
                    self.line += 1;
                    value.push(b'\n');
                }
                Some(b'\n') => {
                    self.line += 1;
                    value.push(b'\n');
                }
                Some(byte) => value.push(byte),
            }
        }

        String::from_utf8(value).map_err(|e| CsvwError::CsvError {
            line: self.line,
            msg: format!("invalid UTF-8 in quoted field: {e}"),
        })
    }

    /// Parse an unquoted field (everything up to the next delimiter or
    /// record terminator).  `first_byte` is the byte that triggered this
    /// branch — it has already been consumed from the stream.
    fn parse_unquoted_field(&mut self, first_byte: u8) -> Result<String, CsvwError> {
        let mut value = vec![first_byte];

        loop {
            match self.peek() {
                None | Some(b'\n') | Some(b'\r') => break,
                Some(peeked) if peeked == self.delimiter => break,
                _ => {
                    if let Some(byte) = self.next_byte() {
                        value.push(byte);
                    }
                }
            }
        }

        String::from_utf8(value).map_err(|e| CsvwError::CsvError {
            line: self.line,
            msg: format!("invalid UTF-8 in field: {e}"),
        })
    }

    // ── record parsing ───────────────────────────────────────────────────────

    /// Check the byte after the end of a just-parsed field and update state.
    ///
    /// Returns `true` when the record has ended (line terminator consumed or
    /// EOF reached).
    fn consume_post_field(&mut self) -> bool {
        match self.peek() {
            Some(peeked) if peeked == self.delimiter => {
                self.pos += 1; // consume delimiter, more fields follow
                false
            }
            Some(b'\r') => {
                self.pos += 1;
                self.skip_cr_if_lf();
                true
            }
            Some(b'\n') => {
                self.pos += 1;
                self.line += 1;
                true
            }
            None => true, // EOF — record ends
            _ => false,   // non-conformant char — tolerate, stay in record
        }
    }

    // ── public API ──────────────────────────────────────────────────────────

    /// Read the next CSV record.
    ///
    /// Returns `Ok(None)` at EOF (including after a trailing newline that
    /// yields no fields).
    pub fn read_record(&mut self) -> Result<Option<CsvRecord>, CsvwError> {
        self.ensure_loaded()?;

        if self.pos >= self.buf.len() {
            return Ok(None);
        }

        let mut fields: Vec<String> = Vec::new();

        loop {
            match self.peek() {
                // EOF: emit whatever we collected (if anything).
                None => break,

                // Record terminator on an empty record — trailing newline.
                Some(b'\r') if fields.is_empty() => {
                    self.pos += 1;
                    self.skip_cr_if_lf();
                    return Ok(None);
                }
                Some(b'\n') if fields.is_empty() => {
                    self.pos += 1;
                    self.line += 1;
                    return Ok(None);
                }

                // Record terminators mid-record (after at least one field).
                Some(b'\r') => {
                    self.pos += 1;
                    self.skip_cr_if_lf();
                    break;
                }
                Some(b'\n') => {
                    self.pos += 1;
                    self.line += 1;
                    break;
                }

                // Delimiter: the *previous* field was empty or we're at the
                // start of the record (meaning first field is empty).
                Some(peeked) if peeked == self.delimiter => {
                    // Push an empty field for the slot before this delimiter.
                    fields.push(String::new());
                    self.pos += 1; // consume the delimiter
                }

                // Quoted field.
                Some(peeked) if peeked == self.quote => {
                    self.pos += 1; // consume opening quote
                    let field = self.parse_quoted_field()?;
                    fields.push(field);
                    if self.consume_post_field() {
                        break;
                    }
                }

                // Unquoted field — `peek()` just returned `Some(_)` so
                // `next_byte()` cannot return `None` here.
                Some(first_peek) => {
                    self.pos += 1; // consume the byte we peeked
                    let field = self.parse_unquoted_field(first_peek)?;
                    fields.push(field);
                    if self.consume_post_field() {
                        break;
                    }
                }
            }
        }

        if fields.is_empty() {
            return Ok(None);
        }

        Ok(Some(CsvRecord { fields }))
    }

    /// Read all records from the source.
    ///
    /// Does NOT strip the header row — call [`parse_csv`] when you need the
    /// header split off automatically.
    pub fn read_all(&mut self) -> Result<Vec<CsvRecord>, CsvwError> {
        let mut records = Vec::new();
        while let Some(record) = self.read_record()? {
            records.push(record);
        }
        Ok(records)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Convenience function
// ────────────────────────────────────────────────────────────────────────────

/// Parse a complete CSV string and split it into a header row plus data rows.
///
/// Returns `(headers, data_records)` where `headers` is the list of column
/// names from the first row and `data_records` contains all subsequent rows.
///
/// # Errors
///
/// Returns [`CsvwError`] if the CSV is syntactically invalid.
pub fn parse_csv(input: &str) -> Result<(Vec<String>, Vec<CsvRecord>), CsvwError> {
    let cursor = std::io::Cursor::new(input.as_bytes());
    let mut reader = CsvReader::new(cursor);

    // First record is the header.
    let header_record = reader.read_record()?.ok_or_else(|| CsvwError::CsvError {
        line: 1,
        msg: "CSV input has no rows".into(),
    })?;
    let headers = header_record.fields;

    // Remaining records are data rows.
    let data = reader.read_all()?;

    Ok((headers, data))
}
