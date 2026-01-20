use crate::xsd::{Boolean, Decimal, Double, Float};
use std::fmt;
use std::num::ParseIntError;
use std::str::FromStr;

/// Error type for values that are too large to be converted to integer
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TooLargeForIntegerError;

impl fmt::Display for TooLargeForIntegerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value is too large for integer")
    }
}

impl std::error::Error for TooLargeForIntegerError {}

/// [XML Schema `integer` datatype](https://www.w3.org/TR/xmlschema11-2/#integer)
///
/// Uses internally a [`i64`].
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
#[repr(transparent)]
pub struct Integer {
    value: i64,
}

impl Integer {
    pub const MAX: Self = Self { value: i64::MAX };
    pub const MIN: Self = Self { value: i64::MIN };

    #[inline]
    #[must_use]
    pub fn from_be_bytes(bytes: [u8; 8]) -> Self {
        Self {
            value: i64::from_be_bytes(bytes),
        }
    }

    #[inline]
    #[must_use]
    pub fn to_be_bytes(self) -> [u8; 8] {
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
        Some(Self {
            value: self.value.checked_mul(rhs.into().value)?,
        })
    }

    /// [op:numeric-integer-divide](https://www.w3.org/TR/xpath-functions-31/#func-numeric-integer-divide)
    ///
    /// Returns `None` in case of division by zero ([FOAR0001](https://www.w3.org/TR/xpath-functions-31/#ERRFOAR0001)) or overflow ([FOAR0002](https://www.w3.org/TR/xpath-functions-31/#ERRFOAR0002)).
    #[inline]
    #[must_use]
    pub fn checked_div(self, rhs: impl Into<Self>) -> Option<Self> {
        Some(Self {
            value: self.value.checked_div(rhs.into().value)?,
        })
    }

    /// [op:numeric-mod](https://www.w3.org/TR/xpath-functions-31/#func-numeric-mod)
    ///
    /// Returns `None` in case of division by zero ([FOAR0001](https://www.w3.org/TR/xpath-functions-31/#ERRFOAR0001)) or overflow ([FOAR0002](https://www.w3.org/TR/xpath-functions-31/#ERRFOAR0002)).
    #[inline]
    #[must_use]
    pub fn checked_rem(self, rhs: impl Into<Self>) -> Option<Self> {
        Some(Self {
            value: self.value.checked_rem(rhs.into().value)?,
        })
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

    /// Checks if the two values are [identical](https://www.w3.org/TR/xmlschema11-2/#identity).
    #[inline]
    #[must_use]
    pub fn is_identical_with(self, other: Self) -> bool {
        self == other
    }
}

impl From<i8> for Integer {
    #[inline]
    fn from(value: i8) -> Self {
        Self {
            value: value.into(),
        }
    }
}

impl From<i16> for Integer {
    #[inline]
    fn from(value: i16) -> Self {
        Self {
            value: value.into(),
        }
    }
}

impl From<i32> for Integer {
    #[inline]
    fn from(value: i32) -> Self {
        Self {
            value: value.into(),
        }
    }
}

impl From<i64> for Integer {
    #[inline]
    fn from(value: i64) -> Self {
        Self { value }
    }
}

impl From<u8> for Integer {
    #[inline]
    fn from(value: u8) -> Self {
        Self {
            value: value.into(),
        }
    }
}

impl From<u16> for Integer {
    #[inline]
    fn from(value: u16) -> Self {
        Self {
            value: value.into(),
        }
    }
}

impl From<u32> for Integer {
    #[inline]
    fn from(value: u32) -> Self {
        Self {
            value: value.into(),
        }
    }
}

impl TryFrom<u64> for Integer {
    type Error = TooLargeForIntegerError;

    #[inline]
    fn try_from(value: u64) -> Result<Self, Self::Error> {
        Ok(Self {
            value: i64::try_from(value).map_err(|_| TooLargeForIntegerError)?,
        })
    }
}

impl TryFrom<i128> for Integer {
    type Error = TooLargeForIntegerError;

    #[inline]
    fn try_from(value: i128) -> Result<Self, Self::Error> {
        Ok(Self {
            value: i64::try_from(value).map_err(|_| TooLargeForIntegerError)?,
        })
    }
}

impl TryFrom<u128> for Integer {
    type Error = TooLargeForIntegerError;

    #[inline]
    fn try_from(value: u128) -> Result<Self, Self::Error> {
        Ok(Self {
            value: i64::try_from(value).map_err(|_| TooLargeForIntegerError)?,
        })
    }
}

impl From<Boolean> for Integer {
    #[inline]
    fn from(value: Boolean) -> Self {
        Self {
            value: if bool::from(value) { 1 } else { 0 },
        }
    }
}

impl From<Integer> for i64 {
    #[inline]
    fn from(value: Integer) -> Self {
        value.value
    }
}

impl FromStr for Integer {
    type Err = ParseIntError;

    #[inline]
    fn from_str(input: &str) -> Result<Self, Self::Err> {
        Ok(i64::from_str(input)?.into())
    }
}

impl fmt::Display for Integer {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.value.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_str() -> Result<(), ParseIntError> {
        assert_eq!(Integer::from_str("123")?.to_string(), "123");
        assert_eq!(Integer::from_str("-456")?.to_string(), "-456");
        assert_eq!(Integer::from_str("0")?.to_string(), "0");
        Ok(())
    }

    #[test]
    fn test_arithmetic_operations() {
        let a = Integer::from(10);
        let b = Integer::from(3);

        assert_eq!(a.checked_add(b).unwrap(), Integer::from(13));
        assert_eq!(a.checked_sub(b).unwrap(), Integer::from(7));
        assert_eq!(a.checked_mul(b).unwrap(), Integer::from(30));
        assert_eq!(a.checked_div(b).unwrap(), Integer::from(3));
        assert_eq!(a.checked_rem(b).unwrap(), Integer::from(1));
    }

    #[test]
    fn test_overflow_handling() {
        let max = Integer::MAX;
        assert!(max.checked_add(Integer::from(1)).is_none());
        
        let min = Integer::MIN;
        assert!(min.checked_sub(Integer::from(1)).is_none());
        assert!(min.checked_neg().is_none());
    }

    #[test]
    fn test_division_by_zero() {
        let a = Integer::from(10);
        let zero = Integer::from(0);
        
        assert!(a.checked_div(zero).is_none());
        assert!(a.checked_rem(zero).is_none());
    }

    #[test]
    fn test_conversions() {
        assert_eq!(Integer::from(42i32), Integer::from(42i64));
        assert_eq!(Integer::from(Boolean::from(true)), Integer::from(1));
        assert_eq!(Integer::from(Boolean::from(false)), Integer::from(0));
    }
}