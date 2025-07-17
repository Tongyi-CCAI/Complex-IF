#!/bin/bash


out_dir=""

model_yaml_path=""
model_yaml_path=""

out_dir=""
model_yaml_path=""


out_dir=$1
model_yaml_path=$2
config_yaml_path=$3

# all
if [ "$config_yaml_path" = "all" ]; then
  config_yaml_paths=("config/evaluator-trace-gpt-gpt.yaml" )
else
  config_yaml_paths=($config_yaml_path)
fi



# 获取操作系统信息
os_type=$(uname)
# 根据操作系统类型执行相应的操作
if [ "$os_type" = "Darwin" ]; then
  eval_on_pai="false"
  echo "当前系统是 macOS."
elif [ "$os_type" = "Linux" ]; then
  eval_on_pai="true"
  echo "当前系统是 Linux."
else
  echo "未知操作系统: $os_type"
fi


type=$(yq e '.type' config/${model_yaml_path})
if [ "$type" = "vllm" ]; then
# if [ "$type" = "hhh" ]; then
  # 开启服务
  served_model_name=$(yq e '.args.name'  config/${model_yaml_path})
  model_path=$(yq e '.args.model_path'  config/${model_yaml_path})
  tensor_parallel_size=$(yq e '.args.tensor_parallel_size'  config/${model_yaml_path})
  echo served_model_name $served_model_name
  echo model_path $model_path
  echo tensor_parallel_size $tensor_parallel_size
  netstat_output=$(netstat -an | grep :8000 | grep LISTEN)
  if [[ -n "${netstat_output}" ]];then
    echo "$served_model_name is ready"
  else
    nohup python -m vllm.entrypoints.openai.api_server --served-model-name ${served_model_name} --model ${model_path} --tensor-parallel-size=${tensor_parallel_size} >& nohup_${served_model_name}.api &
    server_pid=$!
    echo $server_pid > server_pid_${served_model_name}.log
    # 使用netstat命令检查8000端口
    netstat_output=$(netstat -an | grep :8000 | grep LISTEN)
    while true
    do
        sleep 30s
        echo "waiting for local_openai_api server ..."
        netstat_output=$(netstat -an | grep :8000 | grep LISTEN)
        if [[ -n "${netstat_output}" ]];then
            break
        fi
    done
  fi
fi

pids=()
for config_yaml_path in "${config_yaml_paths[@]}";
do
    if [ "$eval_on_pai" = "true" ]; then
      model_yaml_path_pre=$(yq e '.evaluator.model' $config_yaml_path)
      yaml_path_pre="model: !include "$model_yaml_path_pre
      yaml_path_new="model: !include "$model_yaml_path
  #    echo $yaml_path_pre $yaml_path_new
      sed -i "s|$yaml_path_pre|$yaml_path_new|" $config_yaml_path  # linux需要用这个
      sed -i "/out_dir/s|: .*|: ${out_dir}|" "$config_yaml_path"
      mac_dir="/Users/xxx/"
      dsw_dir="xxx"
      sed -i "s|$mac_dir|$dsw_dir|" $config_yaml_path
    else
      yq eval ".evaluator.model = \"${model_yaml_path}\"" -i "${config_yaml_path}"
      yq eval ".evaluator.out_dir = \"${out_dir}\"" -i "${config_yaml_path}"
    fi

    if [ "$type" = "vllm" ]; then
      python evaluate_task.py --config_path ${config_yaml_path} &
      pids+=($!)
      sleep 60
    else
      python evaluate_task.py --config_path ${config_yaml_path}
    fi
done


# 等待所有后台任务完成
for pid in "${pids[@]}"; do
    wait $pid
done

# if [ "$type" = "vllm" ]; then
#   # 关闭服务
#   if ps -p "$server_pid" > /dev/null
#     kill -9 $server_pid
#     sleep 60
#   fi
# fi

sleep 72000
