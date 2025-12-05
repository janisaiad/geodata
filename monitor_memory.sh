#!/bin/bash
# Script pour surveiller l'utilisation de la mémoire GPU et CPU en temps réel

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction pour obtenir l'utilisation GPU
get_gpu_memory() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | while IFS=',' read -r used total util; do
            used=$(echo $used | xargs)
            total=$(echo $total | xargs)
            util=$(echo $util | xargs)
            percent=$((used * 100 / total))
            echo "GPU: ${used}MB/${total}MB (${percent}%) | Util: ${util}%"
        done
    else
        echo "GPU: nvidia-smi not available"
    fi
}

# Fonction pour obtenir l'utilisation CPU/RAM
get_cpu_memory() {
    # RAM
    total_ram=$(free -m | awk '/^Mem:/{print $2}')
    used_ram=$(free -m | awk '/^Mem:/{print $3}')
    available_ram=$(free -m | awk '/^Mem:/{print $7}')
    ram_percent=$((used_ram * 100 / total_ram))
    
    # CPU
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
    
    echo "RAM: ${used_ram}MB/${total_ram}MB (${ram_percent}%) | Available: ${available_ram}MB | CPU: ${cpu_usage}%"
}

# Fonction pour obtenir les processus Python utilisant le plus de mémoire
get_top_processes() {
    echo "Top Python processes:"
    ps aux | grep -E "python.*ablation_studies" | grep -v grep | head -3 | while read line; do
        pid=$(echo $line | awk '{print $2}')
        mem=$(echo $line | awk '{print $6}')
        mem_mb=$((mem / 1024))
        cmd=$(echo $line | awk '{for(i=11;i<=NF;i++) printf "%s ", $i; print ""}')
        echo "  PID $pid: ${mem_mb}MB - ${cmd}"
    done
}

# Boucle principale
clear
echo -e "${BLUE}=== Memory Monitor (Press Ctrl+C to stop) ===${NC}"
echo ""

while true; do
    # Déplacer le curseur au début
    tput cup 1 0
    
    # Afficher la date/heure
    echo -e "${YELLOW}$(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo ""
    
    # GPU Memory
    echo -e "${GREEN}$(get_gpu_memory)${NC}"
    echo ""
    
    # CPU/RAM Memory
    echo -e "${GREEN}$(get_cpu_memory)${NC}"
    echo ""
    
    # Top processes
    echo -e "${BLUE}$(get_top_processes)${NC}"
    
    # Attendre 1 seconde
    sleep 1
done

