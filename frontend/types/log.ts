export interface LogRequest {
  method: string;
  path: string;
  ip: string;
  activity_type: string;
  user_agent: string;
  prompt?: string;
}

export interface LogEntry {
  timestamp: string;
  asctime: string;
  levelname: string;
  name: string;
  message: string;
  request?: LogRequest;
  error?: string;
  duration_ms?: number;
  status_code?: number;
}
