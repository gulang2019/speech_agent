#!/usr/bin/env python3
import ctypes
import ctypes.util
import os


class Extension(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char * 256),
        ("spec_version", ctypes.c_uint32),
    ]


class ApplicationInfo(ctypes.Structure):
    _fields_ = [
        ("s_type", ctypes.c_uint32),
        ("p_next", ctypes.c_void_p),
        ("application_name", ctypes.c_char_p),
        ("application_version", ctypes.c_uint32),
        ("engine_name", ctypes.c_char_p),
        ("engine_version", ctypes.c_uint32),
        ("api_version", ctypes.c_uint32),
    ]


class InstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("s_type", ctypes.c_uint32),
        ("p_next", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("application_info", ctypes.POINTER(ApplicationInfo)),
        ("enabled_layer_count", ctypes.c_uint32),
        ("enabled_layer_names", ctypes.POINTER(ctypes.c_char_p)),
        ("enabled_extension_count", ctypes.c_uint32),
        ("enabled_extension_names", ctypes.POINTER(ctypes.c_char_p)),
    ]


def main() -> None:
    lib_name = ctypes.util.find_library("vulkan") or "libvulkan.so.1"
    lib = ctypes.CDLL(lib_name)
    print(f"loader={lib_name}")

    for icd_name in (
        os.environ.get("VULKAN_PROBE_ICD"),
        "/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0",
        "/.singularity.d/libs/libGLX_nvidia.so.0",
    ):
        if not icd_name:
            continue
        try:
            icd = ctypes.CDLL(icd_name)
            get_proc = icd.vk_icdGetInstanceProcAddr
            get_proc.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            get_proc.restype = ctypes.c_void_p
            pointer = get_proc(None, b"vkCreateInstance")
            print(f"icd={icd_name} vkCreateInstance={pointer}")
            break
        except OSError as exc:
            print(f"icd={icd_name} load_error={exc}")
        except AttributeError as exc:
            print(f"icd={icd_name} symbol_error={exc}")

    get_version = lib.vkEnumerateInstanceVersion
    get_version.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
    get_version.restype = ctypes.c_int32
    version = ctypes.c_uint32(0)
    result = get_version(ctypes.byref(version))
    print(f"vkEnumerateInstanceVersion result={result} version={version.value}")

    get_extensions = lib.vkEnumerateInstanceExtensionProperties
    get_extensions.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(Extension)]
    get_extensions.restype = ctypes.c_int32
    count = ctypes.c_uint32(0)
    result = get_extensions(None, ctypes.byref(count), None)
    print(f"vkEnumerateInstanceExtensionProperties(count) result={result} count={count.value}")
    if result != 0 or count.value == 0:
        return
    entries = (Extension * count.value)()
    result = get_extensions(None, ctypes.byref(count), entries)
    names = [entries[i].name.decode("ascii", "replace") for i in range(count.value)]
    print(f"vkEnumerateInstanceExtensionProperties(data) result={result}")
    print("extensions=" + ",".join(names))

    app_info = ApplicationInfo(
        0,
        None,
        b"robodojo-vulkan-probe",
        1,
        b"none",
        1,
        (1 << 22) | (1 << 12),
    )
    create_info = InstanceCreateInfo(1, None, 0, ctypes.pointer(app_info), 0, None, 0, None)
    create_instance = lib.vkCreateInstance
    create_instance.argtypes = [
        ctypes.POINTER(InstanceCreateInfo),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    create_instance.restype = ctypes.c_int32
    instance = ctypes.c_void_p()
    result = create_instance(ctypes.byref(create_info), None, ctypes.byref(instance))
    print(f"vkCreateInstance result={result} instance={instance.value}")
    if result == 0 and instance.value:
        destroy_instance = lib.vkDestroyInstance
        destroy_instance.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        destroy_instance.restype = None
        destroy_instance(instance, None)
        print("vkDestroyInstance=ok")


if __name__ == "__main__":
    main()
