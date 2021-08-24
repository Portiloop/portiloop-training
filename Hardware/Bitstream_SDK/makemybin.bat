cd portiloop_microblaze/Debug/
mb-objcopy -O binary portiloop_microblaze.elf portiloop_microblaze.bin
cp portiloop_microblaze.bin ..\..\..\Driver\portiloop_microblaze.bin
cp portiloop_microblaze.bin \\192.168.2.99\xilinx\pynq\lib\arduino\portiloop_microblaze.bin
cd ..
cd ..
