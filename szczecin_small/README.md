HOW TO RUN:

1. run `generator.py` (possible to give a scaling factor as a positional argument)
2. run `jtrrouter -n szczecin.net.xml -r flows_file_name.xml -o traffic_file_name.rou.xml --accept-all-destinations` ensuring correct file names
3. create a sumo config file and provide newly generated traffic file name to the `route file value` tag
