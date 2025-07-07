use crate::xsd::{Boolean, Double, Float, Integer, TooLargeForIntegerError};
use std::fmt;
use std::fmt::Write;
use std::str::FromStr;

const DECIMAL_PART_DIGITS: u32 = 18;
const DECIMAL_PART_POW: i128 = 1_000_000_000_000_000_000;
const DECIMAL_PART_POW_MINUS_ONE: i128 = 100_000_000_000_000_000;

/// Error type for decimal parsing
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseDecimalError {
    kind: ParseDecimalErrorKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ParseDecimalErrorKind {
    InvalidFormat,
    Overflow,
}

impl fmt::Display for ParseDecimalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            ParseDecimalErrorKind::InvalidFormat => write!(f, "Invalid decimal format"),
            ParseDecimalErrorKind::Overflow => write!(f, "Decimal overflow"),
        }
    }
}

impl std::error::Error for ParseDecimalError {}

/// Error type for values that are too large for decimal
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TooLargeForDecimalError;

impl fmt::Display for TooLargeForDecimalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value is too large for decimal")
    }
}

impl std::error::Error for TooLargeForDecimalError {}

/// [XML Schema `decimal` datatype](https://www.w3.org/TR/xmlschema11-2/#decimal)
///
/// It stores the decimal in a fix point encoding allowing nearly 18 digits before and 18 digits after ".".
///
/// It stores the value in a [`i128`] integer after multiplying it by 10¹⁸.
#[derive(Eq, PartialEq, Ord, PartialOrd, Debug, Clone, Copy, Hash, Default)]
#[repr(Rust, packed(8))]
pub struct Decimal {
    value: i128, // value * 10^18
}

impl Decimal {
    pub const MAX: Self = Self { value: i128::MAX };
    pub const MIN: Self = Self { value: i128::MIN };
    #[cfg(test)]
    pub const STEP: Self = Self { value: 1 };

    /// Constructs the decimal i / 10^n
    #[inline]
    pub const fn new(i: i128, n: u32) -> Result<Self, TooLargeForDecimalError> {
        let Some(shift) = DECIMAL_PART_DIGITS.checked_sub(n) else {
            return Err(TooLargeForDecimalError);
        };
        let Some(value) = i.checked_mul(10_i128.pow(shift)) else {
            return Err(TooLargeForDecimalError);
        };
        Ok(Self { value })
    }

    pub(crate) const fn new_from_i128_unchecked(value: i128) -> Self {
        Self {
            value: value * DECIMAL_PART_POW,
        }
    }

    #[inline]
    #[must_use]
    pub fn from_be_bytes(bytes: [u8; 16]) -> Self {
        Self {
            value: i128::from_be_bytes(bytes),
        }
    }

    #[inline]
    #[must_use]
    pub fn to_be_bytes(self) -> [u8; 16] {
        self.value.to_be_bytes()
    }

    /// [op:numeric-add](https://www.w3.org/TR/xpath-functions-31/#func-numeric-add)
    ///
    /// Returns `None` in case of overflow ([FOAR0002](https://www.w3.org/TR/xpath-functions-31/#ERRFOAR0002)).
    #[inline]
    #[must_use]
    pub fn checked_add(self, rhs: impl Into<Self>) -> Option<Self> {
        Some(Self {
            value: self.value.checked_add(rhs.into().value)?,
        })
    }

    /// [op:numeric-subtract](https://www.w3.org/TR/xpath-functions-31/#func-numeric-subtract)
    ///
    /// Returns `None` in case of overflow ([FOAR0002](https://www.w3.org/TR/xpath-functions-31/#ERRFOAR0002)).
    #[inline]
    #[must_use]
    pub fn checked_sub(self, rhs: impl Into<Self>) -> Option<Self> {
        Some(Self {
            value: self.value.checked_sub(rhs.into().value)?,
        })
    }

    /// [op:numeric-multiply](https://www.w3.org/TR/xpath-functions-31/#func-numeric-multiply)
    ///
    /// Returns `None` in case of overflow ([FOAR0002](https://www.w3.org/TR/xpath-functions-31/#ERRFOAR0002)).
    #[inline]
    #[must_use]
    pub fn checked_mul(self, rhs: impl Into<Self>) -> Option<Self> {
        self.value
            .checked_div(DECIMAL_PART_POW)?
            .checked_mul(rhs.into().value)
            .map(|value| Self { value })
    }

    /// [op:numeric-divide](https://www.w3.org/TR/xpath-functions-31/#func-numeric-divide)
    ///
    /// Returns `None` in case of division by zero ([FOAR0001](https://www.w3.org/TR/xpath-functions-31/#ERRFOAR0001)) or overflow ([FOAR0002](https://www.w3.org/TR/xpath-functions-31/#ERRFOAR0002)).
    #[inline]
    #[must_use]
    pub fn checked_div(self, rhs: impl Into<Self>) -> Option<Self> {
        let rhs = rhs.into();
        if rhs.value == 0 {
            None
        } else {
            self.value
                .checked_mul(DECIMAL_PART_POW)?
                .checked_div(rhs.value)
                .map(|value| Self { value })
        }
    }

    /// [op:numeric-mod](https://www.w3.org/TR/xpath-functions-31/#func-numeric-mod)
    ///
    /// Returns `None` in case of division by zero ([FOAR0001](https://www.w3.org/TR/xpath-functions-31/#ERRFOAR0001)) or overflow ([FOAR0002](https://www.w3.org/TR/xpath-functions-31/#ERRFOAR0002)).
    #[inline]
    #[must_use]
    pub fn checked_rem(self, rhs: impl Into<Self>) -> Option<Self> {
        let rhs = rhs.into();
        if rhs.value == 0 {
            None
        } else {
            Some(Self {
                value: self.value.checked_rem(rhs.value)?,
            })
        }
    }

    /// [op:numeric-unary-minus](https://www.w3.org/TR/xpath-functions-31/#func-numeric-unary-minus)
    ///
    /// Returns `None` in case of overflow ([FOAR0002](https://www.w3.org/TR/xpath-functions-31/#ERRFOAR0002)).
    #[inline]
    #[must_use]
    pub fn checked_neg(self) -> Option<Self> {
        Some(Self {
            value: self.value.checked_neg()?,
        })
    }

    /// [fn:abs](https://www.w3.org/TR/xpath-functions-31/#func-abs)
    ///
    /// Returns `None` in case of overflow ([FOAR0002](https://www.w3.org/TR/xpath-functions-31/#ERRFOAR0002)).
    #[inline]
    #[must_use]
    pub fn checked_abs(self) -> Option<Self> {
        Some(Self {
            value: self.value.checked_abs()?,
        })
    }

    /// [fn:floor](https://www.w3.org/TR/xpath-functions-31/#func-floor)
    #[inline]
    #[must_use]
    pub fn floor(self) -> Self {
        Self {
            value: if self.value >= 0 {
                (self.value / DECIMAL_PART_POW) * DECIMAL_PART_POW
            } else {
                ((self.value - DECIMAL_PART_POW + 1) / DECIMAL_PART_POW) * DECIMAL_PART_POW
            },
        }
    }

    /// [fn:ceil](https://www.w3.org/TR/xpath-functions-31/#func-ceil)
    #[inline]
    #[must_use]
    pub fn ceil(self) -> Self {
        Self {
            value: if self.value >= 0 {
                ((self.value + DECIMAL_PART_POW - 1) / DECIMAL_PART_POW) * DECIMAL_PART_POW
            } else {
                (self.value / DECIMAL_PART_POW) * DECIMAL_PART_POW
            },
        }
    }

    /// [fn:round](https://www.w3.org/TR/xpath-functions-31/#func-round)
    #[inline]
    #[must_use]
    pub fn round(self) -> Self {
        Self {
            value: if self.value >= 0 {
                ((self.value + DECIMAL_PART_POW / 2) / DECIMAL_PART_POW) * DECIMAL_PART_POW
            } else {
                ((self.value - DECIMAL_PART_POW / 2) / DECIMAL_PART_POW) * DECIMAL_PART_POW
            },
        }
    }

    /// Checks if the two values are [identical](https://www.w3.org/TR/xmlschema11-2/#identity).
    #[inline]
    #[must_use]
    pub fn is_identical_with(self, other: Self) -> bool {
        self == other
    }
}

impl From<Boolean> for Decimal {
    #[inline]
    fn from(value: Boolean) -> Self {
        Self {
            value: if bool::from(value) {
                DECIMAL_PART_POW
            } else {
                0
            },
        }
    }
}

impl From<Integer> for Decimal {
    #[inline]
    fn from(value: Integer) -> Self {
        Self {
            value: i128::from(i64::from(value)) * DECIMAL_PART_POW,
        }
    }
}

impl From<i8> for Decimal {
    #[inline]
    fn from(value: i8) -> Self {
        Self::from(Integer::from(value))
    }
}

impl From<i16> for Decimal {
    #[inline]
    fn from(value: i16) -> Self {
        Self::from(Integer::from(value))
    }
}

impl From<i32> for Decimal {
    #[inline]
    fn from(value: i32) -> Self {
        Self::from(Integer::from(value))
    }
}

impl From<i64> for Decimal {
    #[inline]
    fn from(value: i64) -> Self {
        Self::from(Integer::from(value))
    }
}

impl From<u8> for Decimal {
    #[inline]
    fn from(value: u8) -> Self {
        Self::from(Integer::from(value))
    }
}

impl From<u16> for Decimal {
    #[inline]
    fn from(value: u16) -> Self {
        Self::from(Integer::from(value))
    }
}

impl From<u32> for Decimal {
    #[inline]
    fn from(value: u32) -> Self {
        Self::from(Integer::from(value))
    }
}

impl TryFrom<Decimal> for Integer {
    type Error = TooLargeForIntegerError;

    #[inline]
    fn try_from(value: Decimal) -> Result<Self, Self::Error> {
        i64::try_from(value.value / DECIMAL_PART_POW)
            .map(Integer::from)
            .map_err(|_| TooLargeForIntegerError)
    }
}

impl FromStr for Decimal {
    type Err = ParseDecimalError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        // Remove leading/trailing whitespace
        let input = input.trim();
        
        if input.is_empty() {
            return Err(ParseDecimalError {
                kind: ParseDecimalErrorKind::InvalidFormat,
            });
        }

        // Handle sign
        let (sign, input) = if let Some(input) = input.strip_prefix('-') {
            (-1i128, input)
        } else if let Some(input) = input.strip_prefix('+') {
            (1i128, input)
        } else {
            (1i128, input)
        };

        // Find decimal point
        let (integer_part, fractional_part) = if let Some(dot_pos) = input.find('.') {
            (&input[..dot_pos], &input[dot_pos + 1..])
        } else {
            (input, "")
        };

        // Parse integer part
        let mut value = 0i128;
        for c in integer_part.chars() {
            if !c.is_ascii_digit() {
                return Err(ParseDecimalError {
                    kind: ParseDecimalErrorKind::InvalidFormat,
                });
            }
            value = value
                .checked_mul(10)
                .and_then(|v| v.checked_add((c as u8 - b'0') as i128))
                .ok_or(ParseDecimalError {
                    kind: ParseDecimalErrorKind::Overflow,
                })?;
        }

        // Apply sign
        value = value
            .checked_mul(sign)
            .ok_or(ParseDecimalError {
                kind: ParseDecimalErrorKind::Overflow,
            })?;

        // Scale to decimal part
        value = value
            .checked_mul(DECIMAL_PART_POW)
            .ok_or(ParseDecimalError {
                kind: ParseDecimalErrorKind::Overflow,
            })?;

        // Parse fractional part
        if !fractional_part.is_empty() {
            let mut fractional_value = 0i128;
            let mut scale = DECIMAL_PART_POW_MINUS_ONE;
            
            for c in fractional_part.chars() {
                if !c.is_ascii_digit() {
                    return Err(ParseDecimalError {
                        kind: ParseDecimalErrorKind::InvalidFormat,
                    });
                }
                if scale > 0 {
                    fractional_value += (c as u8 - b'0') as i128 * scale;
                    scale /= 10;
                }
            }

            if sign < 0 {
                fractional_value = -fractional_value;
            }

            value = value
                .checked_add(fractional_value)
                .ok_or(ParseDecimalError {
                    kind: ParseDecimalErrorKind::Overflow,
                })?;
        }

        Ok(Self { value })
    }
}

impl fmt::Display for Decimal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.value == 0 {
            return f.write_char('0');
        }

        let mut value = self.value;
        if value < 0 {
            f.write_char('-')?;
            value = -value;
        }

        let integer_part = value / DECIMAL_PART_POW;
        let fractional_part = value % DECIMAL_PART_POW;

        write!(f, "{integer_part}")?;
        
        if fractional_part != 0 {
            // Remove trailing zeros
            let mut fractional_str = format!("{fractional_part}");
            while fractional_str.ends_with('0') && fractional_str.len() > 1 {
                fractional_str.pop();
            }
            write!(f, ".{fractional_str}")?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_str() -> Result<(), ParseDecimalError> {
        assert_eq!(Decimal::from_str("123")?.to_string(), "123");
        assert_eq!(Decimal::from_str("123.456")?.to_string(), "123.456");
        assert_eq!(Decimal::from_str("-123.456")?.to_string(), "-123.456");
        assert_eq!(Decimal::from_str("0")?.to_string(), "0");
        assert_eq!(Decimal::from_str("0.0")?.to_string(), "0");
        assert_eq!(Decimal::from_str("123.0")?.to_string(), "123");
        Ok(())
    }

    #[test]
    fn test_arithmetic() {
        let a = Decimal::from_str("10.5").unwrap();
        let b = Decimal::from_str("2.5").unwrap();

        assert_eq!(a.checked_add(b).unwrap().to_string(), "13");
        assert_eq!(a.checked_sub(b).unwrap().to_string(), "8");
        assert_eq!(a.checked_div(b).unwrap().to_string(), "4.2");
    }

    #[test]
    fn test_rounding() {
        let a = Decimal::from_str("10.7").unwrap();
        let b = Decimal::from_str("10.3").unwrap();
        let c = Decimal::from_str("-10.7").unwrap();

        assert_eq!(a.floor().to_string(), "10");
        assert_eq!(a.ceil().to_string(), "11");
        assert_eq!(a.round().to_string(), "11");

        assert_eq!(b.floor().to_string(), "10");
        assert_eq!(b.ceil().to_string(), "11");
        assert_eq!(b.round().to_string(), "10");

        assert_eq!(c.floor().to_string(), "-11");
        assert_eq!(c.ceil().to_string(), "-10");
        assert_eq!(c.round().to_string(), "-11");
    }

    #[test]
    fn test_conversions() {
        let dec = Decimal::from(42);
        assert_eq!(dec.to_string(), "42");

        let int_back: Integer = dec.try_into().unwrap();
        assert_eq!(int_back, Integer::from(42));

        let bool_dec = Decimal::from(Boolean::from(true));
        assert_eq!(bool_dec.to_string(), "1");
    }
}