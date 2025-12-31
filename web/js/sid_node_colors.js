/**
 * SID Photography Toolkit - Node Colors
 *
 * Sets standard ComfyUI yellow color theme for all SID_ nodes.
 */

import { app } from "/scripts/app.js";

// Standard ComfyUI yellow swatch colors
const SID_NODE_COLOR = "#432";     // Yellow title bar
const SID_NODE_BGCOLOR = "#653";   // Yellow background

app.registerExtension({
    name: "SID.NodeColors",

    async nodeCreated(node) {
        // Apply to all SID nodes
        if (node.comfyClass && node.comfyClass.startsWith("SID_")) {
            node.color = SID_NODE_COLOR;
            node.bgcolor = SID_NODE_BGCOLOR;
        }
    },

    async loadedGraphNode(node) {
        // Also apply when loading saved workflows
        if (node.comfyClass && node.comfyClass.startsWith("SID_")) {
            node.color = SID_NODE_COLOR;
            node.bgcolor = SID_NODE_BGCOLOR;
        }
    }
});
