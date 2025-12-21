//! Internationalization (i18n) and Localization Support
//!
//! Provides multi-language support for oxirs-chat with translation management,
//! locale detection, and language-specific formatting.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Supported languages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Language {
    English,
    Japanese,
    Spanish,
    French,
    German,
    Chinese,
    Korean,
    Portuguese,
    Russian,
    Arabic,
}

impl Language {
    /// Get language code (ISO 639-1)
    pub fn code(&self) -> &str {
        match self {
            Language::English => "en",
            Language::Japanese => "ja",
            Language::Spanish => "es",
            Language::French => "fr",
            Language::German => "de",
            Language::Chinese => "zh",
            Language::Korean => "ko",
            Language::Portuguese => "pt",
            Language::Russian => "ru",
            Language::Arabic => "ar",
        }
    }

    /// Parse from language code
    pub fn from_code(code: &str) -> Option<Self> {
        match code.to_lowercase().as_str() {
            "en" => Some(Language::English),
            "ja" => Some(Language::Japanese),
            "es" => Some(Language::Spanish),
            "fr" => Some(Language::French),
            "de" => Some(Language::German),
            "zh" | "zh-cn" | "zh-tw" => Some(Language::Chinese),
            "ko" => Some(Language::Korean),
            "pt" => Some(Language::Portuguese),
            "ru" => Some(Language::Russian),
            "ar" => Some(Language::Arabic),
            _ => None,
        }
    }

    /// Get native name of the language
    pub fn native_name(&self) -> &str {
        match self {
            Language::English => "English",
            Language::Japanese => "日本語",
            Language::Spanish => "Español",
            Language::French => "Français",
            Language::German => "Deutsch",
            Language::Chinese => "中文",
            Language::Korean => "한국어",
            Language::Portuguese => "Português",
            Language::Russian => "Русский",
            Language::Arabic => "العربية",
        }
    }

    /// Check if language is right-to-left
    pub fn is_rtl(&self) -> bool {
        matches!(self, Language::Arabic)
    }
}

/// Translation key type
pub type TranslationKey = &'static str;

/// Translation bundle for a specific language
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationBundle {
    language: Language,
    translations: HashMap<String, String>,
}

impl TranslationBundle {
    /// Create a new translation bundle
    pub fn new(language: Language) -> Self {
        let mut bundle = Self {
            language,
            translations: HashMap::new(),
        };

        // Load default translations
        bundle.load_defaults();
        bundle
    }

    /// Load default translations for common strings
    fn load_defaults(&mut self) {
        match self.language {
            Language::English => self.load_english(),
            Language::Japanese => self.load_japanese(),
            Language::Spanish => self.load_spanish(),
            Language::French => self.load_french(),
            Language::German => self.load_german(),
            Language::Chinese => self.load_chinese(),
            Language::Korean => self.load_korean(),
            Language::Portuguese => self.load_portuguese(),
            Language::Russian => self.load_russian(),
            Language::Arabic => self.load_arabic(),
        }
    }

    fn load_english(&mut self) {
        self.translations.extend([
            ("welcome".to_string(), "Welcome to OxiRS Chat".to_string()),
            (
                "error.not_found".to_string(),
                "Resource not found".to_string(),
            ),
            (
                "error.internal".to_string(),
                "Internal server error".to_string(),
            ),
            ("chat.thinking".to_string(), "Thinking...".to_string()),
            (
                "chat.processing".to_string(),
                "Processing your request".to_string(),
            ),
            (
                "chat.searching".to_string(),
                "Searching knowledge graph".to_string(),
            ),
            (
                "sparql.generating".to_string(),
                "Generating SPARQL query".to_string(),
            ),
            (
                "sparql.executing".to_string(),
                "Executing query".to_string(),
            ),
            (
                "results.found".to_string(),
                "Found {count} results".to_string(),
            ),
            (
                "session.created".to_string(),
                "Session created successfully".to_string(),
            ),
            (
                "session.expired".to_string(),
                "Session has expired".to_string(),
            ),
        ]);
    }

    fn load_japanese(&mut self) {
        self.translations.extend([
            ("welcome".to_string(), "OxiRS Chatへようこそ".to_string()),
            (
                "error.not_found".to_string(),
                "リソースが見つかりません".to_string(),
            ),
            (
                "error.internal".to_string(),
                "内部サーバーエラー".to_string(),
            ),
            ("chat.thinking".to_string(), "考え中...".to_string()),
            (
                "chat.processing".to_string(),
                "リクエストを処理中".to_string(),
            ),
            (
                "chat.searching".to_string(),
                "ナレッジグラフを検索中".to_string(),
            ),
            (
                "sparql.generating".to_string(),
                "SPARQLクエリを生成中".to_string(),
            ),
            ("sparql.executing".to_string(), "クエリを実行中".to_string()),
            (
                "results.found".to_string(),
                "{count}件の結果が見つかりました".to_string(),
            ),
            (
                "session.created".to_string(),
                "セッションが作成されました".to_string(),
            ),
            (
                "session.expired".to_string(),
                "セッションの有効期限が切れました".to_string(),
            ),
        ]);
    }

    fn load_spanish(&mut self) {
        self.translations.extend([
            ("welcome".to_string(), "Bienvenido a OxiRS Chat".to_string()),
            (
                "error.not_found".to_string(),
                "Recurso no encontrado".to_string(),
            ),
            (
                "error.internal".to_string(),
                "Error interno del servidor".to_string(),
            ),
            ("chat.thinking".to_string(), "Pensando...".to_string()),
            (
                "chat.processing".to_string(),
                "Procesando su solicitud".to_string(),
            ),
            (
                "chat.searching".to_string(),
                "Buscando en el grafo de conocimiento".to_string(),
            ),
            (
                "sparql.generating".to_string(),
                "Generando consulta SPARQL".to_string(),
            ),
            (
                "sparql.executing".to_string(),
                "Ejecutando consulta".to_string(),
            ),
            (
                "results.found".to_string(),
                "Se encontraron {count} resultados".to_string(),
            ),
            (
                "session.created".to_string(),
                "Sesión creada exitosamente".to_string(),
            ),
            (
                "session.expired".to_string(),
                "La sesión ha expirado".to_string(),
            ),
        ]);
    }

    fn load_french(&mut self) {
        self.translations.extend([
            (
                "welcome".to_string(),
                "Bienvenue sur OxiRS Chat".to_string(),
            ),
            (
                "error.not_found".to_string(),
                "Ressource non trouvée".to_string(),
            ),
            (
                "error.internal".to_string(),
                "Erreur interne du serveur".to_string(),
            ),
            (
                "chat.thinking".to_string(),
                "Réflexion en cours...".to_string(),
            ),
            (
                "chat.processing".to_string(),
                "Traitement de votre demande".to_string(),
            ),
            (
                "chat.searching".to_string(),
                "Recherche dans le graphe de connaissances".to_string(),
            ),
            (
                "sparql.generating".to_string(),
                "Génération de la requête SPARQL".to_string(),
            ),
            (
                "sparql.executing".to_string(),
                "Exécution de la requête".to_string(),
            ),
            (
                "results.found".to_string(),
                "{count} résultats trouvés".to_string(),
            ),
            (
                "session.created".to_string(),
                "Session créée avec succès".to_string(),
            ),
            (
                "session.expired".to_string(),
                "La session a expiré".to_string(),
            ),
        ]);
    }

    fn load_german(&mut self) {
        self.translations.extend([
            (
                "welcome".to_string(),
                "Willkommen bei OxiRS Chat".to_string(),
            ),
            (
                "error.not_found".to_string(),
                "Ressource nicht gefunden".to_string(),
            ),
            (
                "error.internal".to_string(),
                "Interner Serverfehler".to_string(),
            ),
            ("chat.thinking".to_string(), "Denke nach...".to_string()),
            (
                "chat.processing".to_string(),
                "Bearbeite Ihre Anfrage".to_string(),
            ),
            (
                "chat.searching".to_string(),
                "Durchsuche Wissensgraph".to_string(),
            ),
            (
                "sparql.generating".to_string(),
                "Erstelle SPARQL-Abfrage".to_string(),
            ),
            (
                "sparql.executing".to_string(),
                "Führe Abfrage aus".to_string(),
            ),
            (
                "results.found".to_string(),
                "{count} Ergebnisse gefunden".to_string(),
            ),
            (
                "session.created".to_string(),
                "Sitzung erfolgreich erstellt".to_string(),
            ),
            (
                "session.expired".to_string(),
                "Sitzung ist abgelaufen".to_string(),
            ),
        ]);
    }

    fn load_chinese(&mut self) {
        self.translations.extend([
            ("welcome".to_string(), "欢迎使用 OxiRS Chat".to_string()),
            ("error.not_found".to_string(), "未找到资源".to_string()),
            ("error.internal".to_string(), "内部服务器错误".to_string()),
            ("chat.thinking".to_string(), "思考中...".to_string()),
            (
                "chat.processing".to_string(),
                "正在处理您的请求".to_string(),
            ),
            ("chat.searching".to_string(), "正在搜索知识图谱".to_string()),
            (
                "sparql.generating".to_string(),
                "正在生成 SPARQL 查询".to_string(),
            ),
            ("sparql.executing".to_string(), "正在执行查询".to_string()),
            (
                "results.found".to_string(),
                "找到 {count} 个结果".to_string(),
            ),
            ("session.created".to_string(), "会话创建成功".to_string()),
            ("session.expired".to_string(), "会话已过期".to_string()),
        ]);
    }

    fn load_korean(&mut self) {
        self.translations.extend([
            (
                "welcome".to_string(),
                "OxiRS Chat에 오신 것을 환영합니다".to_string(),
            ),
            (
                "error.not_found".to_string(),
                "리소스를 찾을 수 없습니다".to_string(),
            ),
            ("error.internal".to_string(), "내부 서버 오류".to_string()),
            ("chat.thinking".to_string(), "생각 중...".to_string()),
            ("chat.processing".to_string(), "요청 처리 중".to_string()),
            (
                "chat.searching".to_string(),
                "지식 그래프 검색 중".to_string(),
            ),
            (
                "sparql.generating".to_string(),
                "SPARQL 쿼리 생성 중".to_string(),
            ),
            ("sparql.executing".to_string(), "쿼리 실행 중".to_string()),
            (
                "results.found".to_string(),
                "{count}개의 결과를 찾았습니다".to_string(),
            ),
            (
                "session.created".to_string(),
                "세션이 성공적으로 생성되었습니다".to_string(),
            ),
            (
                "session.expired".to_string(),
                "세션이 만료되었습니다".to_string(),
            ),
        ]);
    }

    fn load_portuguese(&mut self) {
        self.translations.extend([
            ("welcome".to_string(), "Bem-vindo ao OxiRS Chat".to_string()),
            (
                "error.not_found".to_string(),
                "Recurso não encontrado".to_string(),
            ),
            (
                "error.internal".to_string(),
                "Erro interno do servidor".to_string(),
            ),
            ("chat.thinking".to_string(), "Pensando...".to_string()),
            (
                "chat.processing".to_string(),
                "Processando sua solicitação".to_string(),
            ),
            (
                "chat.searching".to_string(),
                "Pesquisando grafo de conhecimento".to_string(),
            ),
            (
                "sparql.generating".to_string(),
                "Gerando consulta SPARQL".to_string(),
            ),
            (
                "sparql.executing".to_string(),
                "Executando consulta".to_string(),
            ),
            (
                "results.found".to_string(),
                "{count} resultados encontrados".to_string(),
            ),
            (
                "session.created".to_string(),
                "Sessão criada com sucesso".to_string(),
            ),
            (
                "session.expired".to_string(),
                "A sessão expirou".to_string(),
            ),
        ]);
    }

    fn load_russian(&mut self) {
        self.translations.extend([
            (
                "welcome".to_string(),
                "Добро пожаловать в OxiRS Chat".to_string(),
            ),
            (
                "error.not_found".to_string(),
                "Ресурс не найден".to_string(),
            ),
            (
                "error.internal".to_string(),
                "Внутренняя ошибка сервера".to_string(),
            ),
            ("chat.thinking".to_string(), "Думаю...".to_string()),
            (
                "chat.processing".to_string(),
                "Обработка вашего запроса".to_string(),
            ),
            (
                "chat.searching".to_string(),
                "Поиск в графе знаний".to_string(),
            ),
            (
                "sparql.generating".to_string(),
                "Генерация SPARQL запроса".to_string(),
            ),
            (
                "sparql.executing".to_string(),
                "Выполнение запроса".to_string(),
            ),
            (
                "results.found".to_string(),
                "Найдено {count} результатов".to_string(),
            ),
            (
                "session.created".to_string(),
                "Сессия успешно создана".to_string(),
            ),
            ("session.expired".to_string(), "Сессия истекла".to_string()),
        ]);
    }

    fn load_arabic(&mut self) {
        self.translations.extend([
            ("welcome".to_string(), "مرحبًا بك في OxiRS Chat".to_string()),
            (
                "error.not_found".to_string(),
                "المورد غير موجود".to_string(),
            ),
            (
                "error.internal".to_string(),
                "خطأ داخلي في الخادم".to_string(),
            ),
            ("chat.thinking".to_string(), "جاري التفكير...".to_string()),
            ("chat.processing".to_string(), "معالجة طلبك".to_string()),
            (
                "chat.searching".to_string(),
                "البحث في الرسم البياني للمعرفة".to_string(),
            ),
            (
                "sparql.generating".to_string(),
                "إنشاء استعلام SPARQL".to_string(),
            ),
            ("sparql.executing".to_string(), "تنفيذ الاستعلام".to_string()),
            (
                "results.found".to_string(),
                "تم العثور على {count} نتيجة".to_string(),
            ),
            (
                "session.created".to_string(),
                "تم إنشاء الجلسة بنجاح".to_string(),
            ),
            (
                "session.expired".to_string(),
                "انتهت صلاحية الجلسة".to_string(),
            ),
        ]);
    }

    /// Get translation for a key
    pub fn get(&self, key: &str) -> Option<&str> {
        self.translations.get(key).map(|s| s.as_str())
    }

    /// Get translation with parameters
    pub fn get_with_params(&self, key: &str, params: &HashMap<&str, &str>) -> Option<String> {
        let template = self.get(key)?;
        let mut result = template.to_string();

        for (param_key, param_value) in params {
            result = result.replace(&format!("{{{}}}", param_key), param_value);
        }

        Some(result)
    }

    /// Add or update a translation
    pub fn set(&mut self, key: String, value: String) {
        self.translations.insert(key, value);
    }
}

/// Internationalization manager
pub struct I18nManager {
    bundles: Arc<RwLock<HashMap<Language, TranslationBundle>>>,
    default_language: Language,
}

impl I18nManager {
    /// Create a new i18n manager
    pub fn new(default_language: Language) -> Self {
        let manager = Self {
            bundles: Arc::new(RwLock::new(HashMap::new())),
            default_language,
        };

        info!(
            "Initialized i18n manager with default language: {:?}",
            default_language
        );
        manager
    }

    /// Load all supported languages lazily (async)
    async fn ensure_bundles_loaded(&self) {
        let bundles = self.bundles.read().await;

        // If bundles are already loaded, return early
        if !bundles.is_empty() {
            return;
        }

        // Drop read lock before acquiring write lock
        drop(bundles);

        let mut bundles = self.bundles.write().await;

        // Double-check after acquiring write lock (another thread might have loaded them)
        if !bundles.is_empty() {
            return;
        }

        let languages = [
            Language::English,
            Language::Japanese,
            Language::Spanish,
            Language::French,
            Language::German,
            Language::Chinese,
            Language::Korean,
            Language::Portuguese,
            Language::Russian,
            Language::Arabic,
        ];

        for lang in &languages {
            bundles.insert(*lang, TranslationBundle::new(*lang));
        }

        info!("Loaded {} language bundles", bundles.len());
    }

    /// Get translation for a key in the specified language
    pub async fn translate(&self, language: Language, key: &str) -> String {
        // Ensure bundles are loaded
        self.ensure_bundles_loaded().await;

        let bundles = self.bundles.read().await;

        // Try requested language first
        if let Some(bundle) = bundles.get(&language) {
            if let Some(translation) = bundle.get(key) {
                return translation.to_string();
            }
        }

        // Fall back to default language
        if let Some(bundle) = bundles.get(&self.default_language) {
            if let Some(translation) = bundle.get(key) {
                debug!("Using fallback translation for key: {}", key);
                return translation.to_string();
            }
        }

        // Return key if no translation found
        debug!("No translation found for key: {}", key);
        key.to_string()
    }

    /// Get translation with parameters
    pub async fn translate_with_params(
        &self,
        language: Language,
        key: &str,
        params: HashMap<&str, &str>,
    ) -> String {
        // Ensure bundles are loaded
        self.ensure_bundles_loaded().await;

        let bundles = self.bundles.read().await;

        // Try requested language first
        if let Some(bundle) = bundles.get(&language) {
            if let Some(translation) = bundle.get_with_params(key, &params) {
                return translation;
            }
        }

        // Fall back to default language
        if let Some(bundle) = bundles.get(&self.default_language) {
            if let Some(translation) = bundle.get_with_params(key, &params) {
                return translation;
            }
        }

        // Return key if no translation found
        key.to_string()
    }

    /// Add custom translation
    pub async fn add_translation(
        &self,
        language: Language,
        key: String,
        value: String,
    ) -> Result<()> {
        let mut bundles = self.bundles.write().await;

        let bundle = bundles
            .entry(language)
            .or_insert_with(|| TranslationBundle::new(language));

        bundle.set(key.clone(), value);

        debug!(
            "Added custom translation for language {:?}: {}",
            language, key
        );
        Ok(())
    }

    /// Detect language from Accept-Language header
    pub fn detect_language(accept_language: &str) -> Language {
        // Parse Accept-Language header (simplified)
        let parts: Vec<&str> = accept_language.split(',').collect();

        for part in parts {
            let lang_code = part
                .split(';')
                .next()
                .unwrap_or("")
                .trim()
                .split('-')
                .next()
                .unwrap_or("");

            if let Some(language) = Language::from_code(lang_code) {
                return language;
            }
        }

        Language::English // Default fallback
    }

    /// Get all supported languages
    pub fn supported_languages() -> Vec<Language> {
        vec![
            Language::English,
            Language::Japanese,
            Language::Spanish,
            Language::French,
            Language::German,
            Language::Chinese,
            Language::Korean,
            Language::Portuguese,
            Language::Russian,
            Language::Arabic,
        ]
    }
}

impl Default for I18nManager {
    fn default() -> Self {
        Self::new(Language::English)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_code() {
        assert_eq!(Language::English.code(), "en");
        assert_eq!(Language::Japanese.code(), "ja");
        assert_eq!(Language::Spanish.code(), "es");
    }

    #[test]
    fn test_language_from_code() {
        assert_eq!(Language::from_code("en"), Some(Language::English));
        assert_eq!(Language::from_code("ja"), Some(Language::Japanese));
        assert_eq!(Language::from_code("unknown"), None);
    }

    #[test]
    fn test_language_rtl() {
        assert!(!Language::English.is_rtl());
        assert!(Language::Arabic.is_rtl());
    }

    #[tokio::test]
    async fn test_i18n_manager() {
        let manager = I18nManager::new(Language::English);
        let translation = manager.translate(Language::English, "welcome").await;
        assert!(!translation.is_empty());
    }

    #[tokio::test]
    async fn test_translation_with_params() {
        let manager = I18nManager::new(Language::English);
        let mut params = HashMap::new();
        params.insert("count", "42");

        let translation = manager
            .translate_with_params(Language::English, "results.found", params)
            .await;

        assert!(translation.contains("42"));
    }

    #[test]
    fn test_detect_language() {
        assert_eq!(
            I18nManager::detect_language("ja,en-US;q=0.9"),
            Language::Japanese
        );
        assert_eq!(
            I18nManager::detect_language("fr-FR,fr;q=0.9,en;q=0.8"),
            Language::French
        );
        assert_eq!(I18nManager::detect_language("unknown"), Language::English);
    }
}
