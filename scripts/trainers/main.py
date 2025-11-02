from sklearn.tree import DecisionTreeClassifier

#def main():
 #   print("Hello from trainers!")


#if __name__ == "__main__":
 #   main()
import psutil
import os 

process_info = {p.info["pid"]: p.info["name"] for p in psutil.process_iter(attrs=["pid", "name"])}
name_to_id = dict()

for pid in process_info.keys():
    name = process_info[pid]
    i = len(name_to_id) + 1

    if name not in name_to_id:
        name_to_id[name] = i
    else:
        i = name_to_id[name]
    
    os.system(f"sudo iptables -A OUTPUT -m owner --uid-owner {pid} -j MARK --set-mark {i}  ")
