import jupyter_client
import queue

# files = jupyter_client.find_connection_file()
# print(files)
# specs = jupyter_client.kernelspec.find_kernel_specs()
# print(specs)

km = jupyter_client.KernelManager(kernel_name="clearml")
km.start_kernel()
kc = km.client()
kc.exexute("print('hello world')")
# # 3. 결과 가져오기 (메시지 대기)
# try:
#     while True:
#         msg = kc.get_iopub_msg(timeout=1)
#         print(msg)
#         # if msg['msg_type'] == 'stream':
#         #     print(msg['content']['text'])
#         # elif msg['msg_type'] == 'execute_result':
#         #     print(msg['content']['data'])
# except queue.Empty:
#     pass
kc.stop_channels()
km.shutdown_kernel()