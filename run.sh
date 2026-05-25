#!/usr/bin/env bash
#
# etrader — interactive management script
#
# Two processes are managed independently:
#   • bot      — the trading loop          (python -m src.main)
#   • telegram — the Telegram service      (python -m src.telegram_service)
#
# Run with no arguments to launch the interactive menu, or pass a
# subcommand (see ./run.sh help) to drive everything from CI / scripts.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
DATA_DIR="data"
LOGS_DIR="logs"

BOT_PID_FILE=".run-bot.pid"
TG_PID_FILE=".run-telegram.pid"

BOT_LOG_FILE="$LOGS_DIR/bot.out.log"
TG_LOG_FILE="$LOGS_DIR/telegram.out.log"
TRADER_LOG_FILE="$LOGS_DIR/trader.log"

DEFAULT_CONTROL_HOST="127.0.0.1"
DEFAULT_CONTROL_PORT=8770

# ---------------------------------------------------------------------------
# colors / icons
# ---------------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BLUE='\033[0;34m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

ICON_OK="${GREEN}●${RESET}"
ICON_DOWN="${RED}○${RESET}"
ICON_WARN="${YELLOW}!${RESET}"
ICON_INFO="${CYAN}ℹ${RESET}"
ICON_ARROW="${CYAN}➜${RESET}"
ICON_PLAY="${GREEN}▶${RESET}"
ICON_STOP="${RED}■${RESET}"
ICON_RELOAD="${YELLOW}↻${RESET}"
ICON_BOT="${MAGENTA}◆${RESET}"
ICON_TG="${BLUE}✦${RESET}"

red()    { printf "${RED}%s${RESET}\n" "$*"; }
green()  { printf "${GREEN}%s${RESET}\n" "$*"; }
yellow() { printf "${YELLOW}%s${RESET}\n" "$*"; }
cyan()   { printf "${CYAN}%s${RESET}\n" "$*"; }
bold()   { printf "${BOLD}%s${RESET}\n" "$*"; }
dim()    { printf "${DIM}%s${RESET}\n" "$*"; }

print_header() {
    if [ -t 1 ] && [ -n "${TERM:-}" ] && command -v clear &>/dev/null; then
        clear || true
    fi
    echo -e "${BOLD}${CYAN}"
    echo "  ╔══════════════════════════════════════════════╗"
    echo "  ║              ETRADER  CONTROL                ║"
    echo "  ║       Trading bot + Telegram service         ║"
    echo "  ╚══════════════════════════════════════════════╝"
    echo -e "${RESET}"
}

wait_key() {
    echo ""
    echo -e "  ${DIM}Press any key to return to the menu...${RESET}"
    read -rsn1 || true
}

# ---------------------------------------------------------------------------
# environment / venv
# ---------------------------------------------------------------------------

check_python() {
    if ! command -v python3 &>/dev/null; then
        red "python3 not found. Install Python 3.11+."
        exit 1
    fi
}

ensure_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        red "Virtual environment not found at ./$VENV_DIR — run: ./run.sh setup"
        exit 1
    fi
}

check_env_file() {
    if [ ! -f ".env" ]; then
        yellow "Warning: .env file not found."
        echo "  Populate PUBLIC_KEY / PRIVATE_KEY (+ Azure + Telegram) before starting."
    fi
}

ensure_dirs() {
    mkdir -p "$DATA_DIR" "$LOGS_DIR"
}

# ---------------------------------------------------------------------------
# pid / process helpers
# ---------------------------------------------------------------------------

pid_is_alive() {
    local pid="${1:-}"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

read_pid() {
    cat "$1" 2>/dev/null || true
}

service_pid_file() {
    case "$1" in
        bot)      echo "$BOT_PID_FILE" ;;
        telegram) echo "$TG_PID_FILE" ;;
        *)        echo "" ;;
    esac
}

service_log_file() {
    case "$1" in
        bot)      echo "$BOT_LOG_FILE" ;;
        telegram) echo "$TG_LOG_FILE" ;;
        trader)   echo "$TRADER_LOG_FILE" ;;
        *)        echo "" ;;
    esac
}

service_module() {
    case "$1" in
        bot)      echo "src.main" ;;
        telegram) echo "src.telegram_service" ;;
        *)        echo "" ;;
    esac
}

service_icon() {
    case "$1" in
        bot)      printf "%b" "$ICON_BOT" ;;
        telegram) printf "%b" "$ICON_TG" ;;
        *)        printf " " ;;
    esac
}

service_label() {
    case "$1" in
        bot)      echo "trading bot" ;;
        telegram) echo "Telegram service" ;;
        *)        echo "?" ;;
    esac
}

is_running() {
    local svc="$1" pid_file pid
    pid_file="$(service_pid_file "$svc")"
    [ -n "$pid_file" ] || return 1
    if [ -f "$pid_file" ]; then
        pid="$(read_pid "$pid_file")"
        if pid_is_alive "$pid"; then
            return 0
        fi
        rm -f "$pid_file"
    fi
    return 1
}

stop_pid() {
    local pid="$1"
    local label="$2"

    if ! pid_is_alive "$pid"; then
        return 0
    fi

    echo -e "  ${ICON_STOP} stopping ${BOLD}$label${RESET} (PID $pid)..."
    kill "$pid" 2>/dev/null || true

    local i=0
    while pid_is_alive "$pid" && [ $i -lt 10 ]; do
        sleep 1
        i=$((i + 1))
    done

    if pid_is_alive "$pid"; then
        yellow "  forcefully killing $label (PID $pid)..."
        kill -9 "$pid" 2>/dev/null || true
    fi
}

child_pids() {
    local parent_pid="$1"
    if ! command -v ps &>/dev/null; then
        return 0
    fi
    ps -A -o ppid=,pid= 2>/dev/null \
        | awk -v p="$parent_pid" '$1 == p {print $2}'
}

stop_pid_tree() {
    local pid="$1"
    local label="$2"
    local child

    if ! pid_is_alive "$pid"; then
        return 0
    fi

    for child in $(child_pids "$pid"); do
        stop_pid_tree "$child" "$label child"
    done

    stop_pid "$pid" "$label"
}

# ---------------------------------------------------------------------------
# control api helpers
# ---------------------------------------------------------------------------

control_host() { printf '%s\n' "${ETRADER_CONTROL_HOST:-$DEFAULT_CONTROL_HOST}"; }
control_port() { printf '%s\n' "${ETRADER_CONTROL_PORT:-$DEFAULT_CONTROL_PORT}"; }

control_url() {
    printf 'http://%s:%s%s\n' "$(control_host)" "$(control_port)" "$1"
}

internal_token() {
    if [ -n "${INTERNAL_API_TOKEN:-}" ]; then
        printf '%s\n' "${INTERNAL_API_TOKEN}"
        return
    fi
    # Fallback: pull from .env without sourcing the whole file.
    if [ -f ".env" ]; then
        sed -n 's/^INTERNAL_API_TOKEN=//p' .env | tr -d '"' | tr -d "'" | head -n 1
    fi
}

control_ping() {
    local token
    token="$(internal_token)"
    [ -n "$token" ] || return 1
    local code
    code="$(curl -s -o /dev/null -w '%{http_code}' \
        -H "Authorization: Bearer $token" \
        "$(control_url /ping)" 2>/dev/null || echo "000")"
    [ "$code" = "200" ]
}

# ---------------------------------------------------------------------------
# log helpers
# ---------------------------------------------------------------------------

print_tail() {
    local file="$1"
    local lines="${2:-50}"
    local label="$3"

    if [ ! -f "$file" ]; then
        yellow "  no $label log file yet ($file)."
        return 0
    fi

    bold "$label logs ($file):"
    tail -n "$lines" "$file"
}

# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

cmd_setup() {
    bold "Setting up etrader..."
    check_python

    if [ ! -d "$VENV_DIR" ]; then
        green "Creating Python virtual environment at $VENV_DIR..."
        python3 -m venv "$VENV_DIR"
    fi

    ensure_dirs
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"

    green "Installing requirements.txt..."
    pip install --upgrade pip --quiet
    pip install -r requirements.txt --quiet
    check_env_file

    green "Setup complete."
    echo ""
    echo "  Next steps:"
    echo "    ./run.sh start           # start both bot + telegram"
    echo "    ./run.sh status          # show service health"
}

start_service() {
    local svc="$1"
    local pid_file log_file module label icon
    pid_file="$(service_pid_file "$svc")"
    log_file="$(service_log_file "$svc")"
    module="$(service_module "$svc")"
    label="$(service_label "$svc")"
    icon="$(service_icon "$svc")"

    if [ -z "$pid_file" ] || [ -z "$module" ]; then
        red "unknown service: $svc"
        return 1
    fi

    if is_running "$svc"; then
        yellow "  $label already running (PID $(read_pid "$pid_file"))"
        return 0
    fi

    ensure_venv
    ensure_dirs
    check_env_file

    echo -e "  ${ICON_PLAY} starting ${BOLD}$label${RESET}  $icon  (-> $log_file)"
    nohup "$VENV_DIR/bin/python" -u -m "$module" \
        >> "$log_file" 2>&1 &
    local pid=$!
    disown "$pid" 2>/dev/null || true
    echo "$pid" > "$pid_file"

    sleep 1
    if ! pid_is_alive "$pid"; then
        red "  $label (PID $pid) exited during startup. Last log lines:"
        print_tail "$log_file" 30 "$label"
        rm -f "$pid_file"
        return 1
    fi
    green "  $label started (PID $pid)."
}

stop_service() {
    local svc="$1"
    local pid_file label
    pid_file="$(service_pid_file "$svc")"
    label="$(service_label "$svc")"

    if [ -z "$pid_file" ]; then
        red "unknown service: $svc"
        return 1
    fi

    if is_running "$svc"; then
        stop_pid_tree "$(read_pid "$pid_file")" "$label"
        rm -f "$pid_file"
        green "  $label stopped."
    else
        dim "  $label not running."
        rm -f "$pid_file"
    fi
}

restart_service() {
    local svc="$1"
    echo -e "  ${ICON_RELOAD} restarting ${BOLD}$(service_label "$svc")${RESET}"
    stop_service "$svc"
    sleep 1
    start_service "$svc"
}

cmd_start() {
    local target="${1:-all}"
    case "$target" in
        all|both|"")
            start_service bot
            sleep 2  # give the control HTTP server a moment so telegram lands cleanly
            start_service telegram
            ;;
        bot)      start_service bot ;;
        telegram|tg) start_service telegram ;;
        *)
            red "Unknown start target: $target (use bot | telegram | all)"
            exit 1
            ;;
    esac
}

cmd_stop() {
    local target="${1:-all}"
    case "$target" in
        all|both|"")
            stop_service telegram
            stop_service bot
            ;;
        bot)      stop_service bot ;;
        telegram|tg) stop_service telegram ;;
        *)
            red "Unknown stop target: $target (use bot | telegram | all)"
            exit 1
            ;;
    esac
}

cmd_restart() {
    local target="${1:-all}"
    case "$target" in
        all|both|"")
            cmd_stop all
            sleep 1
            cmd_start all
            ;;
        bot)      restart_service bot ;;
        telegram|tg) restart_service telegram ;;
        *)
            red "Unknown restart target: $target (use bot | telegram | all)"
            exit 1
            ;;
    esac
}

print_service_status() {
    local svc="$1"
    local pid_file label icon
    pid_file="$(service_pid_file "$svc")"
    label="$(service_label "$svc")"
    icon="$(service_icon "$svc")"

    if is_running "$svc"; then
        local pid
        pid="$(read_pid "$pid_file")"
        echo -e "  $icon ${BOLD}$label${RESET}   ${ICON_OK} running  PID=$pid"
        if command -v ps &>/dev/null; then
            ps -p "$pid" -o pid=,etime=,%cpu=,%mem=,command= 2>/dev/null \
                | awk '{printf "      %s\n", $0}'
        fi
    else
        echo -e "  $icon ${BOLD}$label${RESET}   ${ICON_DOWN} stopped"
    fi
}

cmd_status() {
    bold "Services:"
    print_service_status bot
    print_service_status telegram
    echo ""

    bold "Control HTTP API:"
    if control_ping; then
        echo -e "  ${ICON_OK} ${BOLD}$(control_url /ping)${RESET} responding (auth OK)"
    else
        local token
        token="$(internal_token)"
        if [ -z "$token" ]; then
            echo -e "  ${ICON_WARN} no INTERNAL_API_TOKEN in env or .env — control API cannot be probed"
        else
            echo -e "  ${ICON_DOWN} $(control_url /ping) not responding (bot may be stopped or paused)"
        fi
    fi
    echo ""

    bold "Recent activity:"
    if [ -f "$TRADER_LOG_FILE" ]; then
        tail -n 5 "$TRADER_LOG_FILE" | sed 's/^/  /'
    else
        dim "  no trader log yet ($TRADER_LOG_FILE)"
    fi
}

cmd_logs() {
    local target="${1:-trader}"
    local lines="${2:-80}"
    local follow="${3:-false}"
    local file
    file="$(service_log_file "$target")"

    if [ -z "$file" ]; then
        red "Unknown logs target: $target (use bot | telegram | trader)"
        exit 1
    fi

    if [ "$follow" = "true" ]; then
        [ -f "$file" ] || touch "$file"
        bold "Following $file (Ctrl-C to stop)"
        trap ':' INT
        tail -n "$lines" -F "$file" || true
        trap - INT
        echo ""
        yellow "Stopped following logs."
        return 0
    fi

    print_tail "$file" "$lines" "$target"
}

cmd_test() {
    ensure_venv
    bold "Running test suite..."
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    python -m unittest discover -s tests "$@"
}

cmd_clean() {
    bold "Cleaning runtime artifacts..."
    rm -f "$BOT_LOG_FILE" "$TG_LOG_FILE"
    rm -f "$BOT_PID_FILE" "$TG_PID_FILE"
    green "  Removed launch logs and stale PID files."

    if [ -f "$TRADER_LOG_FILE" ]; then
        echo -ne "  ${ICON_ARROW} truncate $TRADER_LOG_FILE? [y/N]: "
        read -r confirm
        case "${confirm:-n}" in
            y|Y|yes|YES) : > "$TRADER_LOG_FILE"; green "  Truncated $TRADER_LOG_FILE" ;;
            *) yellow "  Keeping $TRADER_LOG_FILE" ;;
        esac
    fi

    if [ -f "$DATA_DIR/bot_state.json" ]; then
        echo -ne "  ${ICON_ARROW} delete $DATA_DIR/bot_state.json? [y/N]: "
        read -r confirm
        case "${confirm:-n}" in
            y|Y|yes|YES) rm -f "$DATA_DIR/bot_state.json"; green "  Removed $DATA_DIR/bot_state.json" ;;
            *) yellow "  Keeping $DATA_DIR/bot_state.json" ;;
        esac
    fi
    green "Clean complete."
}

cmd_help() {
    cat <<EOF

$(bold "etrader management script")

Usage: ./run.sh <command> [args...]

$(bold "Service commands (target = bot | telegram | all, default: all):")
  start    [target]          launch the service(s) under nohup in the background
  stop     [target]          stop the service(s)
  restart  [target]          stop + start

$(bold "Inspection commands:")
  status                     show PID, control-API health, last trader log lines
  logs     [target] [n] [f]  tail a log file
                               target: bot | telegram | trader (default: trader)
                               n:      lines (default: 80)
                               f:      "true" to follow (live-tail)

$(bold "Maintenance commands:")
  setup                      create .venv (if missing) and install requirements.txt
  test  [unittest args...]   run python -m unittest discover -s tests
  clean                      remove launch logs / PIDs, optionally truncate trader.log

$(bold "Other:")
  interactive                open the colored menu (default when stdin is a TTY)
  help                       print this list

$(bold "Environment overrides:")
  ETRADER_CONTROL_HOST       host the control API listens on   (default: $DEFAULT_CONTROL_HOST)
  ETRADER_CONTROL_PORT       port the control API listens on   (default: $DEFAULT_CONTROL_PORT)
  INTERNAL_API_TOKEN         bearer token used to probe /ping  (falls back to .env)

$(bold "Examples:")
  ./run.sh setup
  ./run.sh start                    # bot + telegram
  ./run.sh start bot                # only the trading loop
  ./run.sh restart telegram         # bounce just the Telegram poller
  ./run.sh logs bot 200 true        # follow the bot's nohup output
  ./run.sh logs trader 200 true     # follow the rotating trader log
  ./run.sh status                   # quick health overview

EOF
}

# ---------------------------------------------------------------------------
# interactive menu
# ---------------------------------------------------------------------------

# Renders the bot/telegram/both picker and writes the chosen target
# (one of "bot" | "telegram" | "all") into the global $UI_CHOSEN_TARGET.
#
# We deliberately do NOT return the result via stdout: callers used to
# do `tgt="$(ui_choose_target all)"`, which captured the menu lines AND
# the `read` prompt, leaving the user staring at a blank screen while
# the script silently waited for input. Using a global side-channel
# keeps the menu visible on the terminal.
UI_CHOSEN_TARGET=""

ui_choose_target() {
    local default="${1:-all}"
    UI_CHOSEN_TARGET=""
    echo -e "  ${ICON_BOT} ${CYAN}1${RESET})  bot         (trading loop)"
    echo -e "  ${ICON_TG} ${CYAN}2${RESET})  telegram    (Telegram poller)"
    echo -e "  ${CYAN}3${RESET})  both"
    echo ""
    echo -ne "  ${ICON_ARROW} target [$default]: "
    local ans
    read -r ans
    case "${ans:-$default}" in
        1|bot)        UI_CHOSEN_TARGET="bot" ;;
        2|telegram|tg) UI_CHOSEN_TARGET="telegram" ;;
        3|all|both|"") UI_CHOSEN_TARGET="all" ;;
        *)            UI_CHOSEN_TARGET="$default" ;;
    esac
}

ui_logs_menu() {
    print_header
    echo -e "  ${BOLD}Tail Logs${RESET}\n"
    echo -e "  ${ICON_BOT} ${CYAN}1${RESET})  bot.out.log       (stdout/stderr of src.main)"
    echo -e "  ${ICON_TG} ${CYAN}2${RESET})  telegram.out.log  (stdout/stderr of telegram service)"
    echo -e "       ${CYAN}3${RESET})  trader.log        (rotating, structured)"
    echo ""
    echo -ne "  ${ICON_ARROW} log file [3]: "
    read -r choice
    echo -ne "  ${ICON_ARROW} lines [80]: "
    read -r lines
    echo -ne "  ${ICON_ARROW} live-tail (follow)? [y/N]: "
    read -r follow_choice
    echo ""

    local target="trader"
    case "${choice:-3}" in
        1|bot)         target="bot" ;;
        2|telegram|tg) target="telegram" ;;
        3|trader|"")   target="trader" ;;
    esac

    local follow="false"
    case "${follow_choice:-n}" in
        y|Y|yes|YES) follow="true" ;;
    esac

    cmd_logs "$target" "${lines:-80}" "$follow"
    wait_key
}

main_menu() {
    while true; do
        print_header
        echo -e "  ${BOLD}Main Menu${RESET}\n"
        echo -e "  ${DIM}Services${RESET}"
        echo -e "  ${CYAN}1${RESET})  ${ICON_PLAY}  start        (bot / telegram / both)"
        echo -e "  ${CYAN}2${RESET})  ${ICON_STOP}  stop         (bot / telegram / both)"
        echo -e "  ${CYAN}3${RESET})  ${ICON_RELOAD}  restart      (bot / telegram / both)"
        echo -e "  ${CYAN}4${RESET})  ${ICON_INFO}  status"
        echo ""
        echo -e "  ${DIM}Logs / inspection${RESET}"
        echo -e "  ${CYAN}5${RESET})  ${ICON_INFO}  logs"
        echo ""
        echo -e "  ${DIM}Maintenance${RESET}"
        echo -e "  ${CYAN}s${RESET})  setup .venv + deps"
        echo -e "  ${CYAN}t${RESET})  run tests"
        echo -e "  ${CYAN}c${RESET})  clean artifacts"
        echo -e "  ${CYAN}h${RESET})  help"
        echo -e "  ${CYAN}q${RESET})  quit"
        echo ""
        echo -ne "  ${ICON_ARROW} choice: "
        read -r choice

        case "$choice" in
            1)
                print_header
                echo -e "  ${BOLD}${ICON_PLAY} Start${RESET}\n"
                ui_choose_target all
                echo ""
                cmd_start "$UI_CHOSEN_TARGET"
                wait_key
                ;;
            2)
                print_header
                echo -e "  ${BOLD}${ICON_STOP} Stop${RESET}\n"
                ui_choose_target all
                echo ""
                cmd_stop "$UI_CHOSEN_TARGET"
                wait_key
                ;;
            3)
                print_header
                echo -e "  ${BOLD}${ICON_RELOAD} Restart${RESET}\n"
                ui_choose_target all
                echo ""
                cmd_restart "$UI_CHOSEN_TARGET"
                wait_key
                ;;
            4)
                print_header
                echo -e "  ${BOLD}${ICON_INFO} Status${RESET}\n"
                cmd_status
                wait_key
                ;;
            5) ui_logs_menu ;;
            s|S)
                print_header
                echo -e "  ${BOLD}Setup${RESET}\n"
                cmd_setup
                wait_key
                ;;
            t|T)
                print_header
                echo -e "  ${BOLD}Tests${RESET}\n"
                cmd_test
                wait_key
                ;;
            c|C)
                print_header
                echo -e "  ${BOLD}Clean${RESET}\n"
                cmd_clean
                wait_key
                ;;
            h|H)
                print_header
                cmd_help
                wait_key
                ;;
            q|Q)
                echo -e "\n  ${DIM}Goodbye.${RESET}\n"
                exit 0
                ;;
            *)
                echo -e "\n  ${ICON_WARN}  invalid choice"
                sleep 1
                ;;
        esac
    done
}

# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------

run_command() {
    local cmd="${1:-}"
    shift || true

    case "$cmd" in
        setup)         cmd_setup ;;
        start)         cmd_start "${1:-all}" ;;
        stop)          cmd_stop  "${1:-all}" ;;
        restart)       cmd_restart "${1:-all}" ;;
        status)        cmd_status ;;
        logs)          cmd_logs "${1:-trader}" "${2:-80}" "${3:-false}" ;;
        test|tests)    cmd_test "$@" ;;
        clean)         cmd_clean ;;
        help|--help|-h) cmd_help ;;
        interactive)   main_menu ;;
        "")
            if [ -t 0 ] && [ -t 1 ]; then
                main_menu
            else
                yellow "No command given and no interactive terminal detected."
                echo "Try:"
                echo "  ./run.sh start"
                echo "  ./run.sh status"
                echo "  ./run.sh help"
                exit 2
            fi
            ;;
        *)
            red "Unknown command: $cmd"
            cmd_help
            exit 1
            ;;
    esac
}

run_command "${1:-}" "${@:2}"
