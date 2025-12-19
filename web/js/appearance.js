/**
 * SID Photography Toolkit - Node Appearance
 *
 * Sets consistent colors and widths for all SID nodes.
 * Color scheme: Professional blue/purple tones
 */

import { app } from "/scripts/app.js";

const COLOR_THEMES = {
    // All SID nodes - Dark military green
    SID: {
        nodeColor: "#3D4A32",    // Title bar - military green
        nodeBgColor: "#2A3325",  // Body - darker military green
        width: 340
    },
};

const NODE_COLORS = {
    // LLM Provider nodes
    "SID_LLM_API": "SID",
    "SID_LLM_Local": "SID",
    "SID_LLM_Local_API": "SID",

    // Prompt Generator nodes
    "SID_ZImagePromptGeneratorV2": "SID",
    "SID_ZImagePromptGenerator": "SID",  // Legacy node
    "SID_PromptTemplate": "SID",

    // Debug/Testing nodes
    "SID_PromptDebugAgent": "SID",
};

function setNodeColors(node, theme) {
    if (!theme) { return; }
    if (theme.nodeColor) {
        node.color = theme.nodeColor;
    }
    if (theme.nodeBgColor) {
        node.bgcolor = theme.nodeBgColor;
    }
    if (theme.width) {
        node.size = node.size || [140, 80];
        node.size[0] = theme.width;
    }
}

const ext = {
    name: "SID.Photography.appearance",

    nodeCreated(node) {
        const nclass = node.comfyClass;
        if (NODE_COLORS.hasOwnProperty(nclass)) {
            let colorKey = NODE_COLORS[nclass];
            const theme = COLOR_THEMES[colorKey];
            setNodeColors(node, theme);
        }
    }
};

app.registerExtension(ext);
