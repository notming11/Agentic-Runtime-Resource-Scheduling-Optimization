# Implementation of the Load Balancer

class LoadBalancer:
    def __init__(self, servers):
        self.servers = servers
        self.current_server_index = 0

    def get_next_server(self):
        server = self.servers[self.current_server_index]
        self.current_server_index = (self.current_server_index + 1) % len(self.servers)
        return server

    def distribute_load(self, load):
        # Example logic: distribute the load to the servers
        server = self.get_next_server()
        # Simulate sending load to the server
        print(f"Distributing {load} to {server}")

# Example usage
if __name__ == '__main__':
    servers = ['Server1', 'Server2', 'Server3']
    lb = LoadBalancer(servers)
    for i in range(5):
        lb.distribute_load(f'Load-{i+1}')
