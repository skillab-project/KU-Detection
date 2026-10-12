"""
17-KU taxonomy for the KU-Detector-17 model (Elisavet's extension).

These coexist with the legacy K1-K28 taxonomy. In the DB they are stored
with ku_source='detector17' so the two sets never collide.
"""

DETECTOR17_KU_IDS = [
    "MQTT",
    "REST_APIs",
    "Concurrency",
    "WebSockets",
    "Messaging_APIs",
    "OPC_UA",
    "Telemetry_Data",
    "Edge_Computing",
    "OTA_Updates",
    "IoT_Security",
    "EMF",
    "Xtext",
    "Eclipse_RCP",
    "Embedded_RTOS",
    "Matter_Thread",
    "IEC_61499",
    "CMake",
]

NUM_DETECTOR17_KUS = len(DETECTOR17_KU_IDS)   # 17
DETECTOR17_KU_INDEX = {k: i for i, k in enumerate(DETECTOR17_KU_IDS)}

# File extensions this model is trained on (Java + C/C++ + build/config)
DETECTOR17_FILE_TYPES = {
    ".java", ".kt",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx",
    ".py", ".dts", ".dtsi",
    ".cmake", ".xtext", ".ecore",
    ".xml", ".properties",
    ".fbt", ".sys",
}
DETECTOR17_FILE_NAMES = {
    "CMakeLists.txt", "plugin.xml", "MANIFEST.MF",
    "feature.xml", "FreeRTOSConfig.h",
}
