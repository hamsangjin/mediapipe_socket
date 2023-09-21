using System;
using System.Net;
using System.Net.Sockets;
using System.Text;

// 서버의 IP 주소와 포트 번호 설정
string serverIP = "127.0.0.1"; // 서버의 실제 IP 주소로 대체
int serverPort = 25001; // 파이썬 서버와 동일한 포트 번호

// 서버에 연결
TcpClient client = new TcpClient(serverIP, serverPort);
NetworkStream stream = client.GetStream();

// 데이터를 받을 버퍼 설정
byte[] buffer = new byte[102400];
int bytesRead;
// 무한 루프로 데이터 수신 및 출력
while (true)
{
    bytesRead = stream.Read(buffer, 0, buffer.Length);
    string dataReceived = Encoding.UTF8.GetString(buffer, 0, bytesRead);
    Console.WriteLine("------------------\n" + dataReceived);

    // 받은 데이터의 길이가 5보다 작으면 루프 종료
    if (dataReceived.Length < 5)
    {
        Console.WriteLine("웹캠이 종료되었으니 수신 종료");
        break;
    }
}

// 연결 종료
stream.Close();
client.Close();