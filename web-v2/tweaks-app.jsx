// tweaks-app.jsx — mounts the Tweaks panel and applies choices to <html>.
// Used by both index.html and scan.html.

const SKIN_TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "accent": "cobalt",
  "corner": "soft",
  "scale": 100
}/*EDITMODE-END*/;

function applySkinTweaks(t) {
  const root = document.documentElement;
  root.setAttribute("data-accent", t.accent);
  root.setAttribute("data-corner", t.corner);
  root.style.fontSize = (t.scale || 100) + "%";
}

function SkinTweaksApp() {
  const [t, setTweak] = useTweaks(SKIN_TWEAK_DEFAULTS);

  React.useEffect(() => { applySkinTweaks(t); }, [t.accent, t.corner, t.scale]);

  return (
    <TweaksPanel>
      <TweakSection label="Accent direction" />
      <TweakColor
        label="Brand color"
        value={accentHex(t.accent)}
        options={[ACCENT.cobalt, ACCENT.teal, ACCENT.green, ACCENT.warm]}
        onChange={(hex) => setTweak("accent", hexToAccent(hex))}
      />
      <TweakRadio
        label=""
        value={t.accent}
        options={["cobalt", "teal", "green", "warm"]}
        onChange={(v) => setTweak("accent", v)}
      />
      <TweakSection label="Surface" />
      <TweakRadio
        label="Corners"
        value={t.corner}
        options={["soft", "sharp"]}
        onChange={(v) => setTweak("corner", v)}
      />
      <TweakSection label="Type" />
      <TweakSlider
        label="Text scale"
        value={t.scale}
        min={90} max={112} step={2} unit="%"
        onChange={(v) => setTweak("scale", v)}
      />
    </TweaksPanel>
  );
}

const ACCENT = { cobalt: "#3563d6", teal: "#2a8f9e", green: "#3a8f5c", warm: "#d9536a" };
function accentHex(name) { return ACCENT[name] || ACCENT.cobalt; }
function hexToAccent(hex) {
  for (const k in ACCENT) if (ACCENT[k] === hex) return k;
  return "cobalt";
}

ReactDOM.createRoot(document.getElementById("tweaks-root")).render(<SkinTweaksApp />);
