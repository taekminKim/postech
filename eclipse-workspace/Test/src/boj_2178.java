import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Scanner;
import java.util.StringTokenizer;

public class boj_2178 {
	static int arr[][];
	static int visit[][];
	static int dx[] = { 0, 1, 0, -1 };
	static int dy[] = { 1, 0, -1, 0 };
	static int max = 1;
	static int N, M;

	public static void main(String[] args) throws IOException {
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		Scanner sc = new Scanner(System.in);
		StringTokenizer st = new StringTokenizer(br.readLine());
		N = Integer.parseInt(st.nextToken());
		M = Integer.parseInt(st.nextToken());
		arr = new int[N][M];
		visit = new int[N][M];

		for (int i = 0; i < N; i++) {
			String temp = br.readLine();
			for (int j = 0; j < M; j++) {
				arr[i][j] = (int) temp.charAt(j)- 48;
			}
		}

		bfs(0, 0);

		
		System.out.println("test222");
		for(int i=0; i<N; i++) {
			for(int j=0; j<M; j++) {
				System.out.print(visit[i][j]+" ");
			}
			System.out.println();
		}
		
	}

	private static void bfs(int x, int y) {
		visit[0][0] = 1;
		Queue<Point> q = new LinkedList<Point>();
		q.add(new Point(x, y));

		while (!q.isEmpty()) {
			Point temp_p = q.poll();
			System.out.println(temp_p.x + " " + temp_p.y);
			for (int i = 0; i < 4; i++) {
				int nx = temp_p.x + dx[i];
				int ny = temp_p.y + dy[i];
				if(nx >=0 && ny >=0 && nx < N  && ny <M) {
					if(arr[nx][ny]==1 && visit[nx][ny] == 0) {
						q.add(new Point(nx, ny));
						visit[nx][ny] = visit[x][y]+1;
					}
				}
			}
		}
		
		
		
	}
	
	
	
	
}
