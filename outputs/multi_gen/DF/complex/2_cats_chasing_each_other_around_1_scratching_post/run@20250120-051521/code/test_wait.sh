
(
    for i in {1..10}; do
        cmd="sleep 10"
        eval "$cmd"
        sleep 1
        echo $i
        wait
    done
)
