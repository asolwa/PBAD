import sys
import json
import os
import random

INPUT_CSV = "warszawa_traffic.csv"
CONFIG_JSON = "routes_medium.json"
OUTPUT_XML = "warszawa_flows.rou.xml"

def load_route_config(json_file):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        return e

def parse_csv_data(filename):
    flows = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or not line[0].isdigit():
                    continue
                parts = line.split(',')
                if len(parts) >= 2:
                    try:
                        hour = int(parts[0])
                        count = int(parts[1])
                        if count > 0:
                            flows.append({
                                'hour': hour,
                                'count': count,
                                'begin': hour * 3600,
                                'end': (hour + 1) * 3600
                            })
                    except ValueError:
                        continue
    except FileNotFoundError as e:
        return e
    return flows

def split_count_random(total, n_parts):
    if n_parts <= 0: return []
    base = total // n_parts
    remainder = total % n_parts
    counts = [base] * n_parts
    for i in range(remainder):
        counts[i] += 1
    random.shuffle(counts)
    return counts

def split_count_weighted(total, weights):
    total_weight = sum(weights)
    if total_weight == 0:
        return [0] * len(weights)

    counts = []
    current_sum = 0
    
    for w in weights:
        share = int((w / total_weight) * total)
        counts.append(share)
        current_sum += share
    
    remainder = total - current_sum
    
    if remainder > 0:
        indices = list(range(len(weights)))
        random.shuffle(indices)
        
        for i in range(remainder):
            idx = indices[i % len(indices)]
            counts[idx] += 1
            
    return counts

def generate_rou_xml(traffic_data, routes_config, output_name):
    traffic_data.sort(key=lambda x: x['begin'])
    
    start_edges = list(routes_config.keys())
    
    start_weights = []
    for edge in start_edges:
        entry = routes_config[edge]
        if isinstance(entry, dict):
            start_weights.append(entry.get("weight", 1))
        else:
            start_weights.append(1)

    with open(output_name, "w", encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        f.write('    <vType id="standard_car" accel="2.6" decel="4.5" sigma="0.5" length="5.0" minGap="2.5" maxSpeed="50.0"/>\n\n')

        for flow_entry in traffic_data:
            total_cars_in_hour = flow_entry['count']
            hour = flow_entry['hour']
            
            counts_per_start = split_count_weighted(total_cars_in_hour, start_weights)
            
            for i, start_edge in enumerate(start_edges):
                cars_for_this_start = counts_per_start[i]
                
                if cars_for_this_start == 0:
                    continue
                
                config_entry = routes_config[start_edge]
                if isinstance(config_entry, dict):
                    destinations = config_entry.get("destinations", [])
                else:
                    destinations = config_entry
                
                if not destinations:
                    continue

                counts_per_dest = split_count_random(cars_for_this_start, len(destinations))
                
                for j, dest_edge in enumerate(destinations):
                    cars_for_route = counts_per_dest[j]
                    
                    if cars_for_route > 0:
                        flow_id = f"flow_{hour}h_{start_edge}_to_{dest_edge}"
                        
                        f.write(
                            f'    <flow id="{flow_id}" '
                            f'type="standard_car" '
                            f'from="{start_edge}" '
                            f'to="{dest_edge}" '
                            f'begin="{flow_entry["begin"]}" '
                            f'end="{flow_entry["end"]}" '
                            f'number="{cars_for_route}"/>\n'
                        )

        f.write('</routes>')

if __name__ == "__main__":  
    routes_config = load_route_config(CONFIG_JSON)
    if routes_config:
        traffic_data = parse_csv_data(INPUT_CSV)
        
        if traffic_data:
            generate_rou_xml(traffic_data, routes_config, OUTPUT_XML)