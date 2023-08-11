using System;
using System.Net;
using System.Net.Sockets;
using System.Text;

// Console.WriteLine("Hi");

class Program
{
    static void Main(string[] args)
    {
        // 서버의 IP 주소와 포트 번호 설정
        // 로컬 PC IP
        string serverIP = "127.0.0.1"; ]
        // 파이썬 서버와 동일한 포트 번호
        int serverPort = 25001; 
        try
        {
            // 서버에 연결
            // TcpClient 클래스를 사용하여 지정된 IP 주소와 포트 번호에 있는 서버에 연결합니다.
            TcpClient client = new TcpClient(serverIP, serverPort);
            //NetworkStream을 사용하여 데이터를 주고받을 수 있는 스트림을 생성합니다.
            NetworkStream stream = client.GetStream();

            // 데이터를 받을 버퍼 설정
            // 1024 바이트로 다 읽을 수 있을라나 ..
            // 102400 바이트는 되어야 100KB 받음
            byte[] buffer = new byte[1];
            int bytesRead;

            // 무한 루프로 데이터 수신 및 출력
            while (true)
            {
                // stream.Read(buffer, 0, buffer.Length) 호출로 데이터를 읽어옵니다. stream은 서버와의 데이터 통신을 담당하는 스트림입니다.
                // bytesRead 변수에 읽어온 데이터의 바이트 수가 저장됩니다.
                bytesRead = stream.Read(buffer, 0, buffer.Length);

                // Encoding.UTF8.GetString(buffer, 0, bytesRead) 호출을 통해 읽어온 바이트 배열을 문자열로 디코딩합니다.
                // 디코딩된 문자열은 dataReceived 변수에 저장됩니다.
                string dataReceived = Encoding.UTF8.GetString(buffer, 0, bytesRead);
                // Console.WriteLine("Received data:\n" + dataReceived)를 통해 받은 데이터를 콘솔에 출력합니다.
                Console.WriteLine("Received data:\n" + dataReceived);

                // 받은 데이터의 길이가 5보다 작으면 루프 종료
                if (dataReceived.Length < 5)
                {
                    Console.WriteLine("Received data is too short. Exiting loop.");
                    break;
                }
            }

            // 연결 종료
            stream.Close();
            client.Close();
        }
        catch (Exception e)
        {
            // Console.WriteLine("Hi");
            Console.WriteLine("Exception: " + e.Message);
        }
    }
}

Program.Main();