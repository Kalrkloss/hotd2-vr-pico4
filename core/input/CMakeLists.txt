# hotd2-vr: modified in 2026 by mikermak for the Quest VR mode (see "git log master..hotd2-vr").
target_sources(${PROJECT_NAME} PRIVATE
		dreampotato.cpp
		dreampotato.h
        maplelink.cpp
        maplelink.h)
if(NOT LIBRETRO)
    target_sources(${PROJECT_NAME} PRIVATE
            gamepad.h
            gamepad_device.cpp
            gamepad_device.h
            keyboard_device.h
            mapping.cpp
            mapping.h
            mouse.cpp
            mouse.h
            udp_lightgun.cpp
            udp_lightgun.h)
endif()
