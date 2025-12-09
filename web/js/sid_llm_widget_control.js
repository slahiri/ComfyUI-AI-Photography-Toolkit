/**
 * SID Photography Toolkit - LLM Widget Control
 *
 * Disables/enables internal LLM settings when external LLM_MODEL is connected.
 */

import { app } from "/scripts/app.js";

// Fields to disable when external LLM is connected
const LLM_CONTROLLED_WIDGETS = [
    "ai_provider",
    "api_key",
    "model",
    "api_url",
    "max_tokens",
    "temperature"
];

app.registerExtension({
    name: "SID.LLMWidgetControl",

    async nodeCreated(node) {
        // Only apply to SID_ZImagePromptGenerator
        if (node.comfyClass !== "SID_ZImagePromptGenerator") {
            return;
        }

        // Find the llm_model input
        const llmModelInput = node.inputs?.find(input => input.name === "llm_model");
        if (!llmModelInput) {
            return;
        }

        // Store original widget states
        const originalStates = {};

        // Function to update widget states based on connection
        const updateWidgetStates = () => {
            const isConnected = llmModelInput.link !== null;

            if (node.widgets) {
                for (const widget of node.widgets) {
                    if (LLM_CONTROLLED_WIDGETS.includes(widget.name)) {
                        if (isConnected) {
                            // Save original state and disable
                            if (originalStates[widget.name] === undefined) {
                                originalStates[widget.name] = {
                                    disabled: widget.disabled,
                                    origColor: widget.origBackgroundColor
                                };
                            }
                            widget.disabled = true;
                            widget.origBackgroundColor = widget.options?.backgroundColor;
                            if (widget.options) {
                                widget.options.backgroundColor = "#333333";
                            }
                        } else {
                            // Restore original state
                            if (originalStates[widget.name] !== undefined) {
                                widget.disabled = originalStates[widget.name].disabled || false;
                                if (widget.options && originalStates[widget.name].origColor) {
                                    widget.options.backgroundColor = originalStates[widget.name].origColor;
                                }
                            } else {
                                widget.disabled = false;
                            }
                        }
                    }
                }
            }

            // Trigger redraw
            node.setDirtyCanvas(true, true);
        };

        // Override onConnectionsChange to detect llm_model connection changes
        const originalOnConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function(type, index, connected, link_info) {
            if (originalOnConnectionsChange) {
                originalOnConnectionsChange.apply(this, arguments);
            }

            // Check if this is the llm_model input
            if (type === LiteGraph.INPUT) {
                const input = this.inputs[index];
                if (input && input.name === "llm_model") {
                    updateWidgetStates();
                }
            }
        };

        // Initial state check (in case node is loaded with existing connection)
        setTimeout(() => {
            updateWidgetStates();
        }, 100);
    }
});
