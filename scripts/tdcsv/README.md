# TDCSV

The `tdcsv` program stands for *T*cp*D*ump to *CSV* and allows the user to acquire
training data for training machine learning algorithms to classify packets.

Simple enough, right? Let's get started.

### A Universal Example

```sh
./tdcsv --input           data/windows-server-2022.txt \
        --output          7022.csv                     \
        --id              7022                         \
        --max-packets     20000                        \
        --max-packet-size 56                           \
        --deduplicate                                  \
        --noheader
```

This is the command that you will most likely be using, since it does the
following.

- **`--input`**              - find the input `tcpdump` file at `data/windows-server-2022.txt`
- **`--output 7022.csv`**    - output the `csv` file to `./7022.csv`
- **`--id 7022`**            - the `tcpdump` session was run in virtual machine id `7022`
- **`--max_packets 20000`**  - limit the output to 20000 packets
- **`--max-packet-size 56`** - limit the size of a packet payload to 56 bytes
- **`--deduplicate`**        - remove any duplicate packets (after pruning)
- **`--noheader`**           - our program does not require a header.

You may need to change `data/windows-server-2022.txt` to the file you have.

### No Tcpdump .txt File?

This file can be generated using this program as well. Run the
program in `--monitor` mode to automatically listen for packets and at
the end export a `.csv` file using the parameters as described in the 
sample command stated above. The only alteration would be adding the
`--monitor` flag as shown below.

```sh
./tdcsv --monitor         130.208.246.177              \
        --input           data/windows-server-2022.txt \
        --output          7022.csv                     \
        --id              7022                         \
        --max-packets     20000                        \
        --max-packet-size 56                           \
        --deduplicate                                  \
        --noheader
```

This command will be run inside of the hypothetical virtual machine 7022,
which runs a windows server 2022. It's assumed IP is `130.208.246.177.`
It produces an intermediary file by the name of windows-server-2022 which
contains the raw `tcpdump` output, and resides in the data/ directory.
This file can be automatically removed using the `--remove` flag.
All the other options are for generating the `.csv` file from the
(temporary) `windows-server-2022.txt` file.

### Merge CSV files together

To merge multiple `.csv` files, you can use the `merge.sh` utility.

```sh
./merge.sh --output merged.csv *.csv
```

By default, the script will limit each file to 3000 lines. This
behaviour can of course be changed with the `--lines` flag, and
afterwards specifying the maximum amount of lines per file.
