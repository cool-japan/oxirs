//! Total-order floating-point wrappers used by the SPARQL term system.
//!
//! In-house replacement for the `ordered-float` crate, mirroring the subset of
//! its API used in this crate. The wrappers provide the total `Eq`/`Ord`/`Hash`
//! that IEEE-754 floats lack, so parsed literal values can live inside
//! hashable, sortable enums (see [`crate::term`]).
//!
//! Ordering semantics (identical to `ordered_float::OrderedFloat`):
//! - `-inf < ... < -0.0 == +0.0 < ... < +inf < NaN`
//! - every NaN compares equal to every other NaN (payload and sign ignored)
//! - NaN sorts strictly greater than every non-NaN value, including `+inf`
//!
//! This deliberately differs from `f32::total_cmp`/`f64::total_cmp`, which
//! distinguish NaN payloads/signs and order `-0.0 < +0.0`; using `total_cmp`
//! here would make `Ord` disagree with the IEEE-based `Eq` below.
//!
//! Invariant: `Eq`, `Ord`, and `Hash` agree. Values that compare equal hash to
//! the same bits — all NaNs hash to the canonical NaN bit pattern, and zeros
//! are canonicalized to `+0.0` before hashing.

use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

/// `f32` wrapper with a total order: NaN equals NaN and sorts greatest.
#[derive(Clone, Copy, Debug, Default)]
pub struct TotalF32(pub f32);

/// `f64` wrapper with a total order: NaN equals NaN and sorts greatest.
#[derive(Clone, Copy, Debug, Default)]
pub struct TotalF64(pub f64);

macro_rules! impl_total_float {
    ($name:ident, $float:ty) => {
        impl $name {
            /// Return the wrapped primitive float.
            #[inline]
            pub fn into_inner(self) -> $float {
                self.0
            }
        }

        impl PartialEq for $name {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                // IEEE equality (so `-0.0 == +0.0`), extended with
                // `NaN == NaN` to form a proper equivalence relation.
                self.0 == other.0 || (self.0.is_nan() && other.0.is_nan())
            }
        }

        impl Eq for $name {}

        impl PartialOrd for $name {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for $name {
            fn cmp(&self, other: &Self) -> Ordering {
                match self.0.partial_cmp(&other.0) {
                    Some(ordering) => ordering,
                    // `partial_cmp` on primitive floats returns `None` iff at
                    // least one operand is NaN; NaN sorts greater than every
                    // non-NaN value and equal to any other NaN.
                    None => match (self.0.is_nan(), other.0.is_nan()) {
                        (true, true) => Ordering::Equal,
                        (true, false) => Ordering::Greater,
                        _ => Ordering::Less,
                    },
                }
            }
        }

        impl Hash for $name {
            fn hash<H: Hasher>(&self, state: &mut H) {
                // Canonicalize so `Hash` agrees with `Eq`: every NaN hashes as
                // the canonical NaN, and `x + 0.0` maps `-0.0` to `+0.0` while
                // leaving all other values unchanged.
                let bits = if self.0.is_nan() {
                    <$float>::NAN.to_bits()
                } else {
                    (self.0 + 0.0).to_bits()
                };
                bits.hash(state);
            }
        }

        impl From<$float> for $name {
            #[inline]
            fn from(value: $float) -> Self {
                Self(value)
            }
        }

        impl From<$name> for $float {
            #[inline]
            fn from(value: $name) -> Self {
                value.0
            }
        }
    };
}

impl_total_float!(TotalF32, f32);
impl_total_float!(TotalF64, f64);

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;
    use std::collections::HashSet;

    fn hash_of<T: Hash>(value: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn nan_equals_nan_regardless_of_sign_and_payload() {
        assert_eq!(TotalF64(f64::NAN), TotalF64(f64::NAN));
        assert_eq!(TotalF64(f64::NAN), TotalF64(-f64::NAN));
        assert_eq!(TotalF32(f32::NAN), TotalF32(-f32::NAN));
        assert_eq!(
            TotalF64(f64::NAN).cmp(&TotalF64(-f64::NAN)),
            Ordering::Equal
        );
    }

    #[test]
    fn nan_sorts_greater_than_everything() {
        let nan = TotalF64(f64::NAN);
        assert!(nan > TotalF64(f64::INFINITY));
        assert!(nan > TotalF64(f64::MAX));
        assert!(nan > TotalF64(0.0));
        assert!(nan > TotalF64(f64::NEG_INFINITY));
        // Sign of NaN is ignored: a "negative" NaN still sorts greatest.
        assert!(TotalF64(-f64::NAN) > TotalF64(f64::INFINITY));
        assert!(TotalF32(-f32::NAN) > TotalF32(f32::INFINITY));
    }

    #[test]
    fn zero_signs_compare_equal_and_hash_identically() {
        assert_eq!(TotalF64(0.0), TotalF64(-0.0));
        assert_eq!(TotalF64(0.0).cmp(&TotalF64(-0.0)), Ordering::Equal);
        assert_eq!(hash_of(&TotalF64(0.0)), hash_of(&TotalF64(-0.0)));
        assert_eq!(TotalF32(0.0), TotalF32(-0.0));
        assert_eq!(hash_of(&TotalF32(0.0)), hash_of(&TotalF32(-0.0)));
    }

    #[test]
    fn hash_agrees_with_eq_for_nan() {
        assert_eq!(hash_of(&TotalF64(f64::NAN)), hash_of(&TotalF64(-f64::NAN)));
        assert_eq!(hash_of(&TotalF32(f32::NAN)), hash_of(&TotalF32(-f32::NAN)));
        // Plain equal values hash equal too.
        assert_eq!(hash_of(&TotalF64(1.5)), hash_of(&TotalF64(1.5)));
    }

    #[test]
    fn hash_set_deduplicates_equivalent_values() {
        let mut set = HashSet::new();
        set.insert(TotalF64(f64::NAN));
        set.insert(TotalF64(-f64::NAN));
        set.insert(TotalF64(0.0));
        set.insert(TotalF64(-0.0));
        set.insert(TotalF64(1.5));
        // {NaN, 0.0, 1.5}
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn sort_yields_total_order_with_nan_last() {
        let mut values = [
            TotalF64(f64::NAN),
            TotalF64(1.0),
            TotalF64(f64::NEG_INFINITY),
            TotalF64(f64::INFINITY),
            TotalF64(-1.0),
            TotalF64(0.0),
        ];
        values.sort();
        assert_eq!(values[0].0, f64::NEG_INFINITY);
        assert_eq!(values[1].0, -1.0);
        assert_eq!(values[2].0, 0.0);
        assert_eq!(values[3].0, 1.0);
        assert_eq!(values[4].0, f64::INFINITY);
        assert!(values[5].0.is_nan());
    }

    #[test]
    fn non_nan_ordering_matches_ieee() {
        assert!(TotalF64(1.0) < TotalF64(2.0));
        assert!(TotalF64(-2.0) < TotalF64(-1.0));
        assert!(TotalF64(f64::NEG_INFINITY) < TotalF64(f64::MIN));
        assert!(TotalF32(1.0) < TotalF32(2.0));
    }

    #[test]
    fn conversions_round_trip() {
        let wrapped = TotalF64::from(2.5);
        assert_eq!(wrapped.into_inner(), 2.5);
        let raw: f64 = wrapped.into();
        assert_eq!(raw, 2.5);
        let wrapped32 = TotalF32::from(0.25);
        assert_eq!(f32::from(wrapped32), 0.25);
    }
}
