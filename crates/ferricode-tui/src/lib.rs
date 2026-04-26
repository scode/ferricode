//! Terminal UI boundary for Ferricode.
//!
//! This crate may depend on the harness core, but the reverse dependency must
//! never exist. Terminal rendering, input handling, and UI state belong here so
//! the harness can remain usable from the CLI, tests, and future non-terminal
//! front ends.

use ferricode_core::{Harness, HarnessRequest, HarnessResponse};

/// Builds the first TUI-facing response through the core harness.
///
/// The bootstrap implementation returns core output instead of drawing a full
/// screen. That keeps the crate useful for integration tests while
/// preserving the architectural boundary before a real TUI framework is added.
pub fn launch(request: HarnessRequest) -> HarnessResponse {
    Harness::new().handle(&request)
}

#[cfg(test)]
mod tests {
    use ferricode_core::HarnessRequest;

    #[test]
    fn launch_uses_core_harness() {
        let request = HarnessRequest::new("open tui", ".").unwrap();

        let response = super::launch(request);

        assert_eq!(response.summary(), "Received coding task from .: open tui");
    }
}
