import re
import sys

INPUT_FILE = "real_data_small.txt"   
OUTPUT_FILE = "flows"            

def parse_traffic_data(filename):
    data = {}
    current_edge = None

    edge_pattern = re.compile(r"edge\s+(\S+)")
    
    time_pattern = re.compile(r"^\d{2}:00:\s+(\d+)")

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue

                edge_match = edge_pattern.search(line)
                if edge_match:
                    current_edge = edge_match.group(1)
                    if current_edge not in data:
                        data[current_edge] = []
                    continue

                time_match = time_pattern.match(line)
                if time_match and current_edge:
                    count = int(time_match.group(1))
                    data[current_edge].append(count)
                    
    except FileNotFoundError:
        print(f"Error: Could not find file '{filename}'")
        return {}

    return data

def generate_sorted_xml(data, output_name, scale_factor):
    all_flows = []
    
    for edge, counts in data.items():
        for hour, count in enumerate(counts):
            final_count = int(count * scale_factor)
            
            if final_count > 0:
                all_flows.append({
                    'edge': edge,
                    'hour': hour,
                    'count': final_count,
                    'begin': hour * 3600,
                    'end': (hour + 1) * 3600
                })

    all_flows.sort(key=lambda x: x['begin'])

    with open(f"{output_name}_{scale_factor}.xml", "w") as f:
        f.write('<routes>\n')
        for flow in all_flows:
            f.write(
                f'    <flow id="flow_{flow["edge"]}_{flow["hour"]}" '
                f'from="{flow["edge"]}" begin="{flow["begin"]}" '
                f'end="{flow["end"]}" vehsPerHour="{flow["count"]}"/>\n'
            )
        f.write('</routes>')

if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            scale_factor = float(sys.argv[1])
        except ValueError:
            scale_factor = 1.0
    else:
        scale_factor = 1.0

    traffic_data = parse_traffic_data(INPUT_FILE)
    
    if traffic_data:
        generate_sorted_xml(traffic_data, OUTPUT_FILE, scale_factor)