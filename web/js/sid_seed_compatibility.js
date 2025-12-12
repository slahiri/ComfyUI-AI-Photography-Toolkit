/**
 * SID Photography Toolkit - Seed Widget Compatibility
 *
 * Ensures seed widgets in SID nodes are properly detected by GlobalSeed extensions
 * (EasyGlobalSeed, Inspire Pack GlobalSeed, etc.)
 *
 * The issue: V3 API nodes may not expose widgets in the same way as V1 nodes,
 * causing GlobalSeed extensions to miss them when building the seed_widgets map.
 *
 * This extension:
 * 1. Logs debugging info about seed widgets
 * 2. Ensures seed widgets have correct properties
 * 3. Provides a manual test function in console: SID_testSeedWidgets()
 */

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

// List of SID node types that have seed widgets
const SID_SEED_NODES = [
    "SID_ZImagePromptGenerator",
    "SID_ZImagePromptEnhancer",
    "SID_ZImagePhotographyPrompts"
];

// Debug mode - set to true to see detailed logs
const DEBUG = true;

function log(...args) {
    if (DEBUG) {
        console.log("[SID-Seed]", ...args);
    }
}

// Test function - callable from browser console as: SID_testSeedWidgets()
window.SID_testSeedWidgets = function() {
    console.log("=== SID Seed Widget Test ===");

    if (!app.graph || !app.graph._nodes_by_id) {
        console.log("No graph loaded");
        return;
    }

    for (const nodeId in app.graph._nodes_by_id) {
        const node = app.graph._nodes_by_id[nodeId];

        if (SID_SEED_NODES.includes(node.comfyClass)) {
            console.log(`\nNode ${nodeId}: ${node.comfyClass}`);
            console.log("  widgets:", node.widgets);

            if (node.widgets) {
                for (let j = 0; j < node.widgets.length; j++) {
                    const w = node.widgets[j];
                    console.log(`  [${j}] name="${w.name}", type="${w.type}", value=${w.value}`);

                    if (w.name === "seed" || w.name === "noise_seed" || w.name === "seed_num") {
                        console.log(`    ^ This is a seed widget!`);
                        console.log(`    GlobalSeed compatible: ${w.type !== "converted-widget" ? "YES" : "NO (converted-widget)"}`);
                    }
                }
            }

            console.log("  widgets_values:", node.widgets_values);
        }
    }

    // Also check for KSampler for comparison
    for (const nodeId in app.graph._nodes_by_id) {
        const node = app.graph._nodes_by_id[nodeId];
        if (node.comfyClass === "KSampler") {
            console.log(`\nKSampler ${nodeId} for comparison:`);
            if (node.widgets) {
                for (let j = 0; j < node.widgets.length; j++) {
                    const w = node.widgets[j];
                    if (w.name === "seed") {
                        console.log(`  [${j}] name="${w.name}", type="${w.type}", value=${w.value}`);
                    }
                }
            }
        }
    }

    console.log("\n=== End Test ===");
};

app.registerExtension({
    name: "SID.SeedCompatibility",

    async nodeCreated(node) {
        // Only apply to SID nodes with seeds
        if (!SID_SEED_NODES.includes(node.comfyClass)) {
            return;
        }

        log(`Node created: ${node.comfyClass} (id: ${node.id})`);

        // Wait a bit for widgets to be fully initialized
        setTimeout(() => {
            if (!node.widgets) {
                log(`  No widgets found on ${node.comfyClass}`);
                return;
            }

            // Find seed widget and log its properties
            for (let i = 0; i < node.widgets.length; i++) {
                const widget = node.widgets[i];
                if (widget.name === "seed") {
                    log(`  Seed widget found at index ${i}:`);
                    log(`    type: ${widget.type}`);
                    log(`    value: ${widget.value}`);
                    log(`    GlobalSeed compatible: ${widget.type !== "converted-widget"}`);

                    // Ensure the widget type is set correctly for GlobalSeed detection
                    // V3 API widgets might have different type values
                    if (!widget.type || widget.type === "") {
                        log(`    Fixing empty type...`);
                        widget.type = "number";  // Standard widget type for integers
                    }
                }
            }
        }, 200);
    },

    async setup() {
        log("Setting up seed compatibility...");

        // Listen for GlobalSeed events from EasyGlobalSeed
        api.addEventListener("easyuse-global-seed", (event) => {
            log("GlobalSeed event received:", event.detail);

            // Check if our nodes received the seed update
            if (event.detail && event.detail.seed_map) {
                log("Seed map:", event.detail.seed_map);

                for (const nodeId in app.graph._nodes_by_id) {
                    const node = app.graph._nodes_by_id[nodeId];
                    if (SID_SEED_NODES.includes(node.comfyClass)) {
                        const seedValue = event.detail.seed_map[nodeId];
                        if (seedValue !== undefined) {
                            log(`  ${node.comfyClass} (${nodeId}): seed = ${seedValue}`);
                        } else {
                            log(`  ${node.comfyClass} (${nodeId}): NOT in seed_map!`);
                        }
                    }
                }
            }
        });

        log("Seed compatibility extension loaded");
        log("Run SID_testSeedWidgets() in console to test seed widget detection");
    }
});
