/**
 * ImageToPrompt Node Styling
 * Military Green theme: #4B5320
 */

import { app } from "/scripts/app.js";

const MILITARY_GREEN = "#4B5320";
const MILITARY_GREEN_DARK = "#3a4219";

app.registerExtension({
    name: "AIPhotographyToolkit.ImageToPrompt",

    async nodeCreated(node) {
        if (node.comfyClass === "ImageToPrompt") {
            // Set node colors
            node.color = MILITARY_GREEN;
            node.bgcolor = MILITARY_GREEN_DARK;
        }
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ImageToPrompt") {
            // Set default colors for the node type
            nodeType.prototype.color = MILITARY_GREEN;
            nodeType.prototype.bgcolor = MILITARY_GREEN_DARK;
        }
    }
});
