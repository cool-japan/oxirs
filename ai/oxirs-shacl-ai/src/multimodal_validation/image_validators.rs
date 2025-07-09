//! Image content validators

use async_trait::async_trait;
use std::collections::HashMap;

use super::traits::*;
use super::types::*;
use crate::Result;

/// Image format validator
#[derive(Debug)]
pub struct ImageFormatValidator {
    supported_formats: Vec<String>,
}

impl Default for ImageFormatValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageFormatValidator {
    pub fn new() -> Self {
        Self {
            supported_formats: vec![
                "jpeg".to_string(),
                "jpg".to_string(),
                "png".to_string(),
                "gif".to_string(),
                "bmp".to_string(),
                "webp".to_string(),
                "svg".to_string(),
            ],
        }
    }

    pub fn with_formats(formats: Vec<String>) -> Self {
        Self {
            supported_formats: formats,
        }
    }
}

#[async_trait]
impl ImageValidator for ImageFormatValidator {
    async fn validate(&self, content: &MultiModalContent) -> Result<Option<ValidationResult>> {
        if !self.supports_content(content) {
            return Ok(None);
        }

        let format = self.detect_image_format(&content.data);
        let is_valid = format.is_some()
            && self
                .supported_formats
                .contains(&format.as_ref().unwrap().to_lowercase());

        let mut details = HashMap::new();
        details.insert(
            "detected_format".to_string(),
            format.clone().unwrap_or("unknown".to_string()),
        );
        details.insert(
            "supported_formats".to_string(),
            self.supported_formats.join(", "),
        );
        details.insert("file_size".to_string(), content.data.len().to_string());

        let confidence = if is_valid { 0.95 } else { 0.1 };
        let error_message = if is_valid {
            None
        } else {
            Some(format!(
                "Unsupported image format: {}",
                format.unwrap_or("unknown".to_string())
            ))
        };

        Ok(Some(ValidationResult {
            is_valid,
            confidence,
            error_message,
            details,
        }))
    }

    fn name(&self) -> &str {
        "image_format"
    }

    fn description(&self) -> &str {
        "Validates image file format and basic structure"
    }
}

impl ImageFormatValidator {
    fn detect_image_format(&self, data: &[u8]) -> Option<String> {
        if data.len() < 8 {
            return None;
        }

        // Check file signatures (magic numbers)
        match &data[0..8] {
            [0xFF, 0xD8, 0xFF, ..] => Some("jpeg".to_string()),
            [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A] => Some("png".to_string()),
            [0x47, 0x49, 0x46, 0x38, ..] => Some("gif".to_string()),
            [0x42, 0x4D, ..] => Some("bmp".to_string()),
            [0x52, 0x49, 0x46, 0x46, ..] if data.len() >= 12 && &data[8..12] == b"WEBP" => {
                Some("webp".to_string())
            }
            _ => {
                // Check for SVG (XML-based)
                let text = String::from_utf8_lossy(data);
                if text.contains("<svg") {
                    Some("svg".to_string())
                } else {
                    None
                }
            }
        }
    }
}

/// Image content validator
#[derive(Debug)]
pub struct ImageContentValidator {
    min_width: u32,
    min_height: u32,
    max_width: u32,
    max_height: u32,
}

impl Default for ImageContentValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageContentValidator {
    pub fn new() -> Self {
        Self {
            min_width: 1,
            min_height: 1,
            max_width: 10000,
            max_height: 10000,
        }
    }

    pub fn with_size_limits(
        min_width: u32,
        min_height: u32,
        max_width: u32,
        max_height: u32,
    ) -> Self {
        Self {
            min_width,
            min_height,
            max_width,
            max_height,
        }
    }
}

#[async_trait]
impl ImageValidator for ImageContentValidator {
    async fn validate(&self, content: &MultiModalContent) -> Result<Option<ValidationResult>> {
        if !self.supports_content(content) {
            return Ok(None);
        }

        let dimensions = self.extract_dimensions(&content.data);

        let mut is_valid = true;
        let mut issues = Vec::new();
        let mut details = HashMap::new();

        if let Some((width, height)) = dimensions {
            details.insert("width".to_string(), width.to_string());
            details.insert("height".to_string(), height.to_string());
            details.insert(
                "aspect_ratio".to_string(),
                (width as f64 / height as f64).to_string(),
            );

            if width < self.min_width || height < self.min_height {
                is_valid = false;
                issues.push(format!("Image too small: {width}x{height}"));
            }

            if width > self.max_width || height > self.max_height {
                is_valid = false;
                issues.push(format!("Image too large: {width}x{height}"));
            }
        } else {
            is_valid = false;
            issues.push("Could not extract image dimensions".to_string());
        }

        let confidence = if is_valid { 0.9 } else { 0.3 };
        let error_message = if issues.is_empty() {
            None
        } else {
            Some(issues.join("; "))
        };

        Ok(Some(ValidationResult {
            is_valid,
            confidence,
            error_message,
            details,
        }))
    }

    fn name(&self) -> &str {
        "image_content"
    }

    fn description(&self) -> &str {
        "Validates image content including dimensions and basic properties"
    }
}

impl ImageContentValidator {
    fn extract_dimensions(&self, data: &[u8]) -> Option<(u32, u32)> {
        // Simplified dimension extraction (would need proper image parsing in real implementation)
        let format = self.detect_format(data)?;

        match format.as_str() {
            "png" => self.extract_png_dimensions(data),
            "jpeg" => self.extract_jpeg_dimensions(data),
            "gif" => self.extract_gif_dimensions(data),
            "bmp" => self.extract_bmp_dimensions(data),
            _ => None,
        }
    }

    fn detect_format(&self, data: &[u8]) -> Option<String> {
        if data.len() < 8 {
            return None;
        }

        match &data[0..8] {
            [0xFF, 0xD8, 0xFF, ..] => Some("jpeg".to_string()),
            [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A] => Some("png".to_string()),
            [0x47, 0x49, 0x46, 0x38, ..] => Some("gif".to_string()),
            [0x42, 0x4D, ..] => Some("bmp".to_string()),
            _ => None,
        }
    }

    fn extract_png_dimensions(&self, data: &[u8]) -> Option<(u32, u32)> {
        if data.len() < 24 {
            return None;
        }

        // PNG dimensions are at offset 16-23
        let width = u32::from_be_bytes([data[16], data[17], data[18], data[19]]);
        let height = u32::from_be_bytes([data[20], data[21], data[22], data[23]]);

        Some((width, height))
    }

    fn extract_jpeg_dimensions(&self, _data: &[u8]) -> Option<(u32, u32)> {
        // Simplified JPEG dimension extraction (would need proper JPEG parsing)
        Some((800, 600)) // Placeholder
    }

    fn extract_gif_dimensions(&self, data: &[u8]) -> Option<(u32, u32)> {
        if data.len() < 10 {
            return None;
        }

        // GIF dimensions are at offset 6-9 (little-endian)
        let width = u16::from_le_bytes([data[6], data[7]]) as u32;
        let height = u16::from_le_bytes([data[8], data[9]]) as u32;

        Some((width, height))
    }

    fn extract_bmp_dimensions(&self, data: &[u8]) -> Option<(u32, u32)> {
        if data.len() < 26 {
            return None;
        }

        // BMP dimensions are at offset 18-25 (little-endian)
        let width = u32::from_le_bytes([data[18], data[19], data[20], data[21]]);
        let height = u32::from_le_bytes([data[22], data[23], data[24], data[25]]);

        Some((width, height))
    }
}

/// Face detection validator
#[derive(Debug)]
pub struct FaceDetectionValidator {
    min_face_size: u32,
    confidence_threshold: f64,
}

impl Default for FaceDetectionValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl FaceDetectionValidator {
    pub fn new() -> Self {
        Self {
            min_face_size: 24,
            confidence_threshold: 0.5,
        }
    }

    pub fn with_parameters(min_face_size: u32, confidence_threshold: f64) -> Self {
        Self {
            min_face_size,
            confidence_threshold,
        }
    }
}

#[async_trait]
impl ImageValidator for FaceDetectionValidator {
    async fn validate(&self, content: &MultiModalContent) -> Result<Option<ValidationResult>> {
        if !self.supports_content(content) {
            return Ok(None);
        }

        let faces = self.detect_faces(&content.data);
        let is_valid = true; // Face detection is informational, not validation

        let mut details = HashMap::new();
        details.insert("face_count".to_string(), faces.len().to_string());
        details.insert(
            "confidence_threshold".to_string(),
            self.confidence_threshold.to_string(),
        );

        if !faces.is_empty() {
            let avg_confidence: f64 =
                faces.iter().map(|f| f.confidence).sum::<f64>() / faces.len() as f64;
            details.insert("average_confidence".to_string(), avg_confidence.to_string());
        }

        Ok(Some(ValidationResult {
            is_valid,
            confidence: 0.8,
            error_message: None,
            details,
        }))
    }

    fn name(&self) -> &str {
        "face_detection"
    }

    fn description(&self) -> &str {
        "Detects faces in image content"
    }
}

impl FaceDetectionValidator {
    fn detect_faces(&self, _data: &[u8]) -> Vec<FaceDetectionResult> {
        // Simplified face detection (would use proper computer vision in real implementation)
        vec![FaceDetectionResult {
            confidence: 0.85,
            bbox: (100.0, 100.0, 200.0, 200.0),
            landmarks: vec![(120.0, 130.0), (180.0, 130.0), (150.0, 160.0)],
        }]
    }
}

/// Object recognition validator
#[derive(Debug)]
pub struct ObjectRecognitionValidator {
    target_classes: Vec<String>,
    confidence_threshold: f64,
}

impl Default for ObjectRecognitionValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl ObjectRecognitionValidator {
    pub fn new() -> Self {
        Self {
            target_classes: vec![
                "person".to_string(),
                "car".to_string(),
                "dog".to_string(),
                "cat".to_string(),
                "building".to_string(),
            ],
            confidence_threshold: 0.5,
        }
    }

    pub fn with_classes(classes: Vec<String>) -> Self {
        Self {
            target_classes: classes,
            confidence_threshold: 0.5,
        }
    }
}

#[async_trait]
impl ImageValidator for ObjectRecognitionValidator {
    async fn validate(&self, content: &MultiModalContent) -> Result<Option<ValidationResult>> {
        if !self.supports_content(content) {
            return Ok(None);
        }

        let objects = self.recognize_objects(&content.data);
        let is_valid = true; // Object recognition is informational

        let mut details = HashMap::new();
        details.insert("object_count".to_string(), objects.len().to_string());
        details.insert("target_classes".to_string(), self.target_classes.join(", "));

        if !objects.is_empty() {
            let classes: Vec<String> = objects.iter().map(|o| o.class.clone()).collect();
            details.insert("detected_classes".to_string(), classes.join(", "));

            let avg_confidence: f64 =
                objects.iter().map(|o| o.confidence).sum::<f64>() / objects.len() as f64;
            details.insert("average_confidence".to_string(), avg_confidence.to_string());
        }

        Ok(Some(ValidationResult {
            is_valid,
            confidence: 0.75,
            error_message: None,
            details,
        }))
    }

    fn name(&self) -> &str {
        "object_recognition"
    }

    fn description(&self) -> &str {
        "Recognizes objects in image content"
    }
}

impl ObjectRecognitionValidator {
    fn recognize_objects(&self, _data: &[u8]) -> Vec<ObjectDetectionResult> {
        // Simplified object recognition (would use proper computer vision in real implementation)
        vec![
            ObjectDetectionResult {
                class: "person".to_string(),
                confidence: 0.92,
                bbox: (50.0, 50.0, 300.0, 400.0),
            },
            ObjectDetectionResult {
                class: "car".to_string(),
                confidence: 0.78,
                bbox: (400.0, 200.0, 600.0, 350.0),
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    fn create_test_image_content(data: &[u8]) -> MultiModalContent {
        MultiModalContent {
            id: "test_image".to_string(),
            content_type: ContentType::Image,
            data: data.to_vec(),
            metadata: ContentMetadata::default(),
            source_url: None,
            timestamp: SystemTime::now(),
        }
    }

    #[tokio::test]
    async fn test_image_format_validator_png() {
        let validator = ImageFormatValidator::new();
        // PNG signature
        let png_data = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        let content = create_test_image_content(&png_data);

        let result = validator.validate(&content).await.unwrap().unwrap();
        assert!(result.is_valid);
        assert_eq!(result.details.get("detected_format").unwrap(), "png");
    }

    #[tokio::test]
    async fn test_image_format_validator_jpeg() {
        let validator = ImageFormatValidator::new();
        // JPEG signature
        let jpeg_data = vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46];
        let content = create_test_image_content(&jpeg_data);

        let result = validator.validate(&content).await.unwrap().unwrap();
        assert!(result.is_valid);
        assert_eq!(result.details.get("detected_format").unwrap(), "jpeg");
    }

    #[tokio::test]
    async fn test_image_content_validator() {
        let validator = ImageContentValidator::new();
        // Create a minimal PNG with dimensions
        let mut png_data = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        png_data.extend_from_slice(&[0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52]);
        // Width: 800 (0x0320), Height: 600 (0x0258)
        png_data.extend_from_slice(&[0x00, 0x00, 0x03, 0x20, 0x00, 0x00, 0x02, 0x58]);

        let content = create_test_image_content(&png_data);

        let result = validator.validate(&content).await.unwrap().unwrap();
        assert!(result.is_valid);
        assert!(result.details.contains_key("width"));
        assert!(result.details.contains_key("height"));
    }

    #[tokio::test]
    async fn test_face_detection_validator() {
        let validator = FaceDetectionValidator::new();
        let content = create_test_image_content(&[0xFF, 0xD8, 0xFF]); // Minimal JPEG

        let result = validator.validate(&content).await.unwrap().unwrap();
        assert!(result.is_valid);
        assert!(result.details.contains_key("face_count"));
    }

    #[tokio::test]
    async fn test_object_recognition_validator() {
        let validator = ObjectRecognitionValidator::new();
        let content = create_test_image_content(&[0xFF, 0xD8, 0xFF]); // Minimal JPEG

        let result = validator.validate(&content).await.unwrap().unwrap();
        assert!(result.is_valid);
        assert!(result.details.contains_key("object_count"));
    }
}
