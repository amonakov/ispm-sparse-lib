#ifndef ISPM_INITIALIZE_H
#define ISPM_INITIALIZE_H

extern void ispm_initialize();

struct DeviceIDString {
  char str[32];
};

extern DeviceIDString get_device_string();

#endif
