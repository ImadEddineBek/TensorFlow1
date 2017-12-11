# import socket
# # import threading
# # import socketserver
# #
# # class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
# #
# #     def handle(self):
# #         data = self.request.recv(1024)
# #         cur_thread = threading.current_thread()
# #         response = "{}: {}".format(cur_thread.name, data)
# #         self.request.sendall(b'worked')
# #
# # class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
# #     pass
# #
# # def client(ip, port, message):
# #     sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# #     sock.connect((ip, port))
# #     try:
# #         sock.sendall(message)
# #         response = sock.recv(1024)
# #         print("Received: {}".format(response))
# #     finally:
# #         sock.close()
# #
# # if __name__ == "__main__":
# ing Completed in: ', total
import tensorflow as tf
print(tf.__version__)




