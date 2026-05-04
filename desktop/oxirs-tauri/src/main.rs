#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod canbus;
mod chat;
mod error;
mod query_builder;

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            chat::send_message,
            chat::list_sessions,
            chat::create_session,
            chat::get_session_history,
            query_builder::generate_sparql,
            query_builder::validate_sparql,
            query_builder::parse_sparql_to_graph,
            query_builder::get_example_queries,
            canbus::get_frames,
            canbus::get_bus_stats,
            canbus::lookup_pgn,
            canbus::get_pgn_database,
            canbus::clear_frames,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
