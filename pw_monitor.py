"""Point a sounddevice input stream at the PipeWire monitor of system output.

The monitor of a sink can't be opened as an ALSA device by name: PortAudio
(under sounddevice) builds its device list from ALSA alone and has no
PulseAudio backend, so PulseAudio-style source names like
`alsa_output.<card>.analog-stereo.monitor` never appear in `query_devices()`.
The ALSA "pipewire" device does exist, but it follows the default *source* --
the microphone.

What does work is the pipewire ALSA plugin's own target, taken from
$PIPEWIRE_NODE: capturing against a *sink* makes PipeWire link to that sink's
monitor ports. Two sharp edges, both found the hard way:

  - This client build accepts only a numeric node id there. A node *name* is
    silently ignored and you are quietly handed the microphone instead -- which
    still shows plenty of signal when speakers are playing, so it looks like it
    worked. Hence `sink_node_id()` and a hard failure when it finds nothing.
  - Node ids are assigned at runtime and change across reboots and device
    changes, so the id has to be resolved at launch rather than written down.
"""

import json
import subprocess


def sink_node_id(name_hint=None):
    """The PipeWire node id of a sink whose monitor carries system output.

    With `name_hint`, the first sink whose node.name contains it; otherwise the
    current default sink, falling back to the only/first sink present. Returns
    (id, node_name), or None if nothing matched.
    """
    try:
        out = subprocess.run(
            ["pw-dump"], capture_output=True, text=True, timeout=10, check=True
        ).stdout
        dump = json.loads(out)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None  # no pw-dump, no PipeWire, or output we don't understand

    sinks = {}        # node.name -> node id
    default_name = None
    for obj in dump:
        props = (obj.get("info") or {}).get("props") or {}
        if props.get("media.class") == "Audio/Sink" and props.get("node.name"):
            sinks[props["node.name"]] = obj["id"]
        # The "default" metadata object holds the user's chosen sink, by name.
        if (obj.get("props") or {}).get("metadata.name") == "default":
            for entry in obj.get("metadata") or []:
                if entry.get("key") == "default.audio.sink":
                    value = entry.get("value")
                    default_name = (
                        value.get("name") if isinstance(value, dict) else value
                    )

    if not sinks:
        return None
    if name_hint:
        for node_name, node_id in sorted(sinks.items()):
            if name_hint in node_name:
                return node_id, node_name
        return None
    if default_name in sinks:
        return sinks[default_name], default_name
    node_name = sorted(sinks)[0]
    return sinks[node_name], node_name


def describe_sinks():
    """The available sink names, for an error message."""
    try:
        dump = json.loads(
            subprocess.run(
                ["pw-dump"], capture_output=True, text=True, timeout=10, check=True
            ).stdout
        )
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return []
    return sorted(
        props["node.name"]
        for props in ((o.get("info") or {}).get("props") or {} for o in dump)
        if props.get("media.class") == "Audio/Sink" and props.get("node.name")
    )
