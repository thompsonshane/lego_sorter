import socket               # Import socket module

counter = 0
s = socket.socket()         # Create a socket object
host = socket.gethostname()        # Get local machine name
port = 12345                # Reserve a port for your service.
print (host)                # I wanted to see the hostname
s.bind((host, port))        # Bind to the port

s.listen(5)                 # Now wait for client connection.
print ("Waiting for client")
while True:
    c, addr = s.accept()     # Establish connection with client.
    print ('Got connection from', addr)
    print ("Receiving...")
    f = open('datasets/from_pi/brick_' + str(counter) + '.jpg', 'wb')
    l = c.recv(1024)
    while l:
        f.write(l)
        l = c.recv(1024)
    f.close()
    print ("Done Receiving")
    c.close()
    counter += 1

print ("All done")