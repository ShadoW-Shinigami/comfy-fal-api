import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// ── localStorage helpers ────────────────────────────────────────────────────

const KEYS_STORAGE = "fal_api_keys";
const ACTIVE_STORAGE = "fal_active_key_name";

function loadKeys() {
  try {
    return JSON.parse(localStorage.getItem(KEYS_STORAGE)) || {};
  } catch {
    return {};
  }
}

function saveKeys(keys) {
  localStorage.setItem(KEYS_STORAGE, JSON.stringify(keys));
}

function getActiveKeyName() {
  return localStorage.getItem(ACTIVE_STORAGE) || "";
}

function setActiveKeyName(name) {
  localStorage.setItem(ACTIVE_STORAGE, name);
}

// ── Send key to server ──────────────────────────────────────────────────────

async function pushActiveKeyToServer() {
  const name = getActiveKeyName();
  const keys = loadKeys();
  const key = keys[name];
  if (!key) return;
  try {
    await api.fetchApi("/fal-api/set-key", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key, name }),
    });
  } catch (e) {
    console.error("[FAL Key Manager] Failed to push key to server:", e);
  }
}

// ── Queue-prompt hook ───────────────────────────────────────────────────────

let hooked = false;
function installQueueHook() {
  if (hooked) return;
  hooked = true;
  const origQueue = app.queuePrompt.bind(app);
  app.queuePrompt = async function (...args) {
    await pushActiveKeyToServer();
    return origQueue(...args);
  };
}

// ── Manage-keys dialog ──────────────────────────────────────────────────────

function showManageKeysDialog(node) {
  const overlay = document.createElement("div");
  Object.assign(overlay.style, {
    position: "fixed",
    inset: "0",
    background: "rgba(0,0,0,0.5)",
    zIndex: "100000",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  });

  const dialog = document.createElement("div");
  Object.assign(dialog.style, {
    background: "#1e1e1e",
    color: "#ccc",
    borderRadius: "8px",
    padding: "20px",
    minWidth: "380px",
    maxWidth: "480px",
    fontFamily: "sans-serif",
    fontSize: "13px",
    boxShadow: "0 8px 32px rgba(0,0,0,0.6)",
  });

  function render() {
    const keys = loadKeys();
    const names = Object.keys(keys);

    dialog.innerHTML = "";

    // Title
    const title = document.createElement("h3");
    title.textContent = "Manage FAL API Keys";
    Object.assign(title.style, { margin: "0 0 14px", color: "#fff" });
    dialog.appendChild(title);

    // Existing keys list
    if (names.length) {
      const list = document.createElement("div");
      Object.assign(list.style, { marginBottom: "14px" });
      for (const n of names) {
        const row = document.createElement("div");
        Object.assign(row.style, {
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "6px 8px",
          marginBottom: "4px",
          background: "#2a2a2a",
          borderRadius: "4px",
        });

        const info = document.createElement("span");
        const preview = keys[n].slice(0, 8) + "...";
        info.textContent = `${n}  (${preview})`;
        row.appendChild(info);

        const del = document.createElement("button");
        del.textContent = "Remove";
        Object.assign(del.style, {
          background: "#c44",
          color: "#fff",
          border: "none",
          borderRadius: "4px",
          padding: "3px 10px",
          cursor: "pointer",
        });
        del.onclick = () => {
          const updated = loadKeys();
          delete updated[n];
          saveKeys(updated);
          if (getActiveKeyName() === n) setActiveKeyName("");
          render();
          refreshNodeWidgets(node);
        };
        row.appendChild(del);
        list.appendChild(row);
      }
      dialog.appendChild(list);
    } else {
      const empty = document.createElement("p");
      empty.textContent = "No keys saved yet.";
      Object.assign(empty.style, { color: "#888", marginBottom: "14px" });
      dialog.appendChild(empty);
    }

    // Add-key form
    const form = document.createElement("div");
    Object.assign(form.style, {
      display: "flex",
      gap: "6px",
      marginBottom: "14px",
    });

    const nameInput = document.createElement("input");
    nameInput.placeholder = "Name";
    Object.assign(nameInput.style, {
      flex: "1",
      padding: "6px",
      background: "#333",
      color: "#fff",
      border: "1px solid #555",
      borderRadius: "4px",
    });

    const keyInput = document.createElement("input");
    keyInput.type = "password";
    keyInput.placeholder = "fal-xxxx...";
    Object.assign(keyInput.style, {
      flex: "2",
      padding: "6px",
      background: "#333",
      color: "#fff",
      border: "1px solid #555",
      borderRadius: "4px",
    });

    const addBtn = document.createElement("button");
    addBtn.textContent = "Add";
    Object.assign(addBtn.style, {
      background: "#4a4",
      color: "#fff",
      border: "none",
      borderRadius: "4px",
      padding: "6px 14px",
      cursor: "pointer",
    });
    addBtn.onclick = () => {
      const n = nameInput.value.trim();
      const k = keyInput.value.trim();
      if (!n || !k) return;
      const updated = loadKeys();
      updated[n] = k;
      saveKeys(updated);
      render();
      refreshNodeWidgets(node);
    };

    form.appendChild(nameInput);
    form.appendChild(keyInput);
    form.appendChild(addBtn);
    dialog.appendChild(form);

    // Close button
    const closeBtn = document.createElement("button");
    closeBtn.textContent = "Close";
    Object.assign(closeBtn.style, {
      background: "#555",
      color: "#fff",
      border: "none",
      borderRadius: "4px",
      padding: "6px 20px",
      cursor: "pointer",
      display: "block",
      marginLeft: "auto",
    });
    closeBtn.onclick = () => overlay.remove();
    dialog.appendChild(closeBtn);
  }

  render();
  overlay.appendChild(dialog);
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) overlay.remove();
  });
  document.body.appendChild(overlay);
}

// ── Refresh combo widget on the node ────────────────────────────────────────

function refreshNodeWidgets(node) {
  if (!node) return;
  const keys = loadKeys();
  const names = Object.keys(keys);
  const options = names.length ? names : ["(no keys)"];

  const combo = node.widgets?.find((w) => w.name === "key_selector");
  if (combo) {
    combo.options.values = options;
    const active = getActiveKeyName();
    combo.value = names.includes(active) ? active : options[0];
  }
  node.setDirtyCanvas(true, true);
}

// ── Register the extension ──────────────────────────────────────────────────

app.registerExtension({
  name: "fal.ApiKeyManager",

  async setup() {
    installQueueHook();

    // Listen for status events from the server node
    api.addEventListener("fal-key-status", (event) => {
      const data = event.detail;
      const nodes = app.graph._nodes.filter(
        (n) => n.type === "FalApiKeyManager"
      );
      for (const node of nodes) {
        const w = node.widgets?.find((w) => w.name === "status_display");
        if (w) {
          w.value = `Active: ${data.active_key_name}`;
          node.setDirtyCanvas(true, true);
        }
      }
    });
  },

  async nodeCreated(node) {
    if (node.comfyClass !== "FalApiKeyManager") return;

    const keys = loadKeys();
    const names = Object.keys(keys);
    const options = names.length ? names : ["(no keys)"];
    const active = getActiveKeyName();

    // Key selector combo
    const combo = node.addWidget(
      "combo",
      "key_selector",
      names.includes(active) ? active : options[0],
      function (value) {
        if (value && value !== "(no keys)") {
          setActiveKeyName(value);
        }
      },
      { values: options }
    );

    // Status display
    node.addWidget("text", "status_display", `Active: ${active || "default"}`, () => {}, {
      multiline: false,
      serialize: false,
    });

    // Manage Keys button
    node.addWidget("button", "manage_keys", "Manage Keys", () => {
      showManageKeysDialog(node);
    });

    // Push on first load if a key is already selected
    if (active && keys[active]) {
      pushActiveKeyToServer();
    }

    node.serialize_widgets = false;
  },
});
