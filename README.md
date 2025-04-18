# Using pyRAPL
```sudo chmod -R a+r /sys/class/powercap/intel-rapl``` or ```sudo chown -R energy /sys/class/powercap/intel-rapl```

# Set Sys Host
```
ip route | grep default
```

# Socket Failed to Connect or Send
If the performance of the physical computer is not great, the control system may fail to pause and unpause the container within a reasonable time, resulting into failure of socket communication. Try reduce the number of containers instead.