/**
 * SID Prompt Template Widget Extension
 *
 * Automatically populates the prompt text area when a template is selected.
 */

import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "SID.PromptTemplateWidget",

    async nodeCreated(node) {
        if (node.comfyClass !== "SID_PromptTemplate") {
            return;
        }

        // Find the template and prompt_text widgets
        const templateWidget = node.widgets?.find(w => w.name === "template");
        const promptWidget = node.widgets?.find(w => w.name === "prompt_text");

        if (!templateWidget || !promptWidget) {
            console.log("[SID] Could not find template or prompt_text widgets");
            return;
        }

        // Store original callback
        const originalCallback = templateWidget.callback;

        // Override callback to fetch template content
        templateWidget.callback = async function(value) {
            // Call original callback if exists
            if (originalCallback) {
                originalCallback.call(this, value);
            }

            // Fetch template content from API
            try {
                const response = await fetch(`/sid/api/template/${encodeURIComponent(value)}`);
                if (response.ok) {
                    const data = await response.json();
                    if (data.system) {
                        promptWidget.value = data.system;
                        // Trigger widget update
                        if (promptWidget.callback) {
                            promptWidget.callback(data.system);
                        }
                        // Request canvas redraw
                        app.graph.setDirtyCanvas(true, true);
                    }
                } else {
                    console.warn("[SID] Failed to fetch template:", value);
                }
            } catch (error) {
                console.error("[SID] Error fetching template:", error);
            }
        };

        // Load initial template content if prompt is empty
        if (!promptWidget.value && templateWidget.value) {
            templateWidget.callback(templateWidget.value);
        }
    }
});
