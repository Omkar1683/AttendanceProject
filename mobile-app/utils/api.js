import AsyncStorage from "@react-native-async-storage/async-storage";
import * as FileSystem from "expo-file-system";
import * as Sharing from "expo-sharing";

// Backend server URL - ensure your mobile device is on the same network
const BASE_URL = "http://192.168.0.107:5000";
const WS_URL = "ws://192.168.0.107:5000"; // WebSocket base (same host, ws:// scheme)

const getHeaders = async () => {
  const token = await AsyncStorage.getItem("userToken");
  const headers = {
    "Content-Type": "application/json",
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  return headers;
};

export const api = {
  // Auth
  login: async (email, password) => {
    try {
      const response = await fetch(`${BASE_URL}/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const data = await response.json();
      if (data.status === "success") {
        await AsyncStorage.setItem("userToken", data.token);
        await AsyncStorage.setItem("userInfo", JSON.stringify(data.user));
        return data;
      } else {
        throw new Error(data.message || "Login failed");
      }
    } catch (error) {
      throw error;
    }
  },

  logout: async () => {
    await AsyncStorage.removeItem("userToken");
    await AsyncStorage.removeItem("userInfo");
  },

  getUserInfo: async () => {
    const info = await AsyncStorage.getItem("userInfo");
    return info ? JSON.parse(info) : null;
  },

  // Data Fetching
  getClasses: async (teacherId) => {
    const headers = await getHeaders();
    const response = await fetch(
      `${BASE_URL}/classes?teacher_id=${teacherId}`,
      { headers },
    );
    return response.json();
  },

  // ── Queue-based scanning pipeline ─────────────────────────────────────────

  /**
   * Fire-and-forget frame enqueue.
   * Posts a camera frame to the backend queue and returns immediately.
   * Recognition happens asynchronously in a background worker.
   *
   * @param {string} frameUri   - Local URI from expo-camera takePictureAsync()
   * @param {string} sessionId  - Active session ID
   * @returns {Promise<{status: 'queued'|'full'|'error'}>}
   */
  enqueueFrame: async (frameUri, sessionId) => {
    const form = new FormData();
    form.append("file", {
      uri: frameUri,
      type: "image/jpeg",
      name: "frame.jpg",
    });
    form.append("session_id", String(sessionId));
    try {
      const res = await fetch(`${BASE_URL}/scan/enqueue`, {
        method: "POST",
        body: form,
        // Do NOT set Content-Type manually — fetch sets multipart boundary
      });
      return res.json();
    } catch {
      return { status: "error" };
    }
  },

  /**
   * Poll the backend queue counters (fallback when WebSocket is unavailable).
   * @returns {Promise<{queued, processing, completed, failed}>}
   */
  getQueueStatus: async (sessionId = "") => {
    try {
      const headers = await getHeaders();
      const query = sessionId ? `?session_id=${sessionId}` : "";
      const res = await fetch(`${BASE_URL}/queue/status${query}`, { headers });
      return res.json();
    } catch {
      return {
        queued: 0,
        processing: 0,
        completed: 0,
        failed: 0,
        present_count: 0,
        marked_students: [],
      };
    }
  },

  /**
   * Open a WebSocket connection and join a session room.
   * The backend (Flask-SocketIO) will push `attendance_update` events.
   *
   * @param {string}   sessionId   - Active session ID to subscribe to
   * @param {function} onUpdate    - Called with {student_name, present_count, worker, ...}
   * @param {function} onStatus    - Called with {queued, processing, completed, failed}
   * @returns {{ close: function }} - Call .close() to disconnect
   */
  connectSocket: (sessionId, onUpdate, onStatus) => {
    const wsUrl = `${WS_URL}/socket.io/?EIO=4&transport=websocket`;
    let ws;
    let closed = false;

    const connect = () => {
      if (closed) return;
      ws = new WebSocket(wsUrl);

      const sendJoin = () => {
        try {
          ws.send(`42["join_session",{"session_id":"${sessionId}"}]`);
        } catch (_) { }
      };

      ws.onopen = () => {
        console.log("[Socket] Connected");
        sendJoin();
      };

      ws.onmessage = (event) => {
        const msg = event.data;
        // Engine.IO / Socket.IO handshake packet ('0' or '40')
        if (
          typeof msg === "string" &&
          (msg.startsWith("0") || msg.startsWith("40"))
        ) {
          sendJoin();
          return;
        }
        // Socket.IO heartbeat ping: respond with pong
        if (msg === "2") {
          ws.send("3");
          return;
        }
        // Data messages start with '42'
        if (typeof msg === "string" && msg.startsWith("42")) {
          try {
            const [eventName, payload] = JSON.parse(msg.slice(2));
            if (eventName === "attendance_update" && onUpdate)
              onUpdate(payload);
            if (eventName === "queue_status" && onStatus) onStatus(payload);
          } catch {
            /* ignore malformed */
          }
        }
      };

      ws.onclose = () => {
        if (!closed) {
          console.log("[Socket] Disconnected — reconnecting in 2s");
          setTimeout(connect, 2000);
        }
      };

      ws.onerror = (err) => {
        console.warn("[Socket] Error:", err.message);
      };
    };

    connect();

    return {
      close: () => {
        closed = true;
        ws && ws.close();
      },
    };
  },

  // ── Create Subject/Class ──────────────────────────────────────────────────────
  createClass: async ({
    name,
    code,
    student_ids = [],
    batch,
    department,
    schedule,
  }) => {
    const headers = await getHeaders();
    // total_students is derived from selected students count (or 1 minimum)
    const total_students = Math.max(student_ids.length, 1);
    const response = await fetch(`${BASE_URL}/classes/create`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        name,
        code,
        total_students,
        student_ids,
        batch,
        department,
        schedule,
      }),
    });
    return response.json();
  },

  // ── Update existing class enrollment & batch ────────────────────────────────
  updateClass: async (classId, { students, batch }) => {
    const headers = await getHeaders();
    const response = await fetch(`${BASE_URL}/classes/${classId}`, {
      method: "PUT",
      headers,
      body: JSON.stringify({ students, batch }),
    });
    return response.json();
  },

  // ── Get all students (for class assignment picker) ────────────────────────────
  getStudents: async (batch = "") => {
    const headers = await getHeaders();
    const query = batch ? `?batch=${encodeURIComponent(batch)}` : "";
    const response = await fetch(`${BASE_URL}/students${query}`, { headers });
    return response.json();
  },

  getTodayAnalytics: async (classId) => {
    const headers = await getHeaders();
    const response = await fetch(
      `${BASE_URL}/analytics/today?class_id=${classId}`,
      { headers },
    );
    return response.json();
  },

  getDefaulters: async (classId, threshold = 75) => {
    const headers = await getHeaders();
    const response = await fetch(
      `${BASE_URL}/analytics/defaulters?class_id=${classId}&threshold=${threshold}`,
      { headers },
    );
    return response.json();
  },

  // Session Management
  createSession: async (classId, location) => {
    const headers = await getHeaders();
    const response = await fetch(`${BASE_URL}/sessions/create`, {
      method: "POST",
      headers,
      body: JSON.stringify({ class_id: classId, location }),
    });
    return response.json();
  },

  stopSession: async (sessionId) => {
    const headers = await getHeaders();
    const response = await fetch(`${BASE_URL}/sessions/stop`, {
      method: "POST",
      headers,
      body: JSON.stringify({ session_id: sessionId }),
    });
    return response.json();
  },

  // Reports
  getClassReport: async (classId, month, year) => {
    const headers = await getHeaders();
    const response = await fetch(
      `${BASE_URL}/reports/class?class_id=${classId}&month=${month}&year=${year}`,
      { headers },
    );
    return response.json();
  },

  getStudentReport: async (studentId) => {
    const headers = await getHeaders();
    const response = await fetch(`${BASE_URL}/reports/student/${studentId}`, {
      headers,
    });
    return response.json();
  },

  // ── Student self-service ─────────────────────────────────────────────────────────
  getStudentProfile: async () => {
    const headers = await getHeaders();
    const response = await fetch(`${BASE_URL}/student/profile`, { headers });
    return response.json();
  },

  updateStudentProfile: async (data) => {
    const headers = await getHeaders();
    const response = await fetch(`${BASE_URL}/student/profile`, {
      method: "PUT",
      headers,
      body: JSON.stringify(data),
    });
    return response.json();
  },

  changePassword: async (old_password, new_password) => {
    const headers = await getHeaders();
    const response = await fetch(`${BASE_URL}/student/change-password`, {
      method: "POST",
      headers,
      body: JSON.stringify({ old_password, new_password }),
    });
    return response.json();
  },

  // ── NEW: Manual Attendance Edit ─────────────────────────────────────────────────
  manualAttendance: async (studentId, sessionId, status) => {
    const headers = await getHeaders();
    const response = await fetch(`${BASE_URL}/attendance/manual`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        student_id: studentId,
        session_id: sessionId,
        status, // 'Present' or 'Absent'
      }),
    });
    return response.json();
  },

  // ── NEW: Attendance by specific date (YYYY-MM-DD) ───────────────────────────────
  // Returns: { status, date, sessions: [...], students: [{student_id, name, roll_no, status}] }
  getAttendanceByDate: async (classId, dateStr) => {
    const headers = await getHeaders();
    const response = await fetch(
      `${BASE_URL}/attendance/by-date?class_id=${classId}&date=${dateStr}`,
      { headers },
    );
    return response.json();
  },

  // ── CSV Download & Share ──────────────────────────────────────────────────────────
  // SDK 54+: uses new File class from expo-file-system/next.
  // Old APIs (downloadAsync, writeAsStringAsync) are fully deprecated in SDK 54.
  downloadAndShareCSV: async (classId, month, year, className) => {
    const token = await AsyncStorage.getItem("userToken");

    if (!token) {
      throw new Error("Authentication token missing. Please log in again.");
    }
    if (!classId) {
      throw new Error("Class ID is required to export the report.");
    }

    const url = `${BASE_URL}/reports/export-csv?class_id=${classId}&month=${month}&year=${year}`;
    const safeClassName = (className || "report").replace(/\s+/g, "_");
    const fileName = `attendance_${safeClassName}_${month}_${year}.csv`;

    // ── Step 1: Fetch CSV from backend ────────────────────────────────────
    let response;
    try {
      response = await fetch(url, {
        method: "GET",
        headers: {
          Authorization: `Bearer ${token}`,
          Accept: "text/csv, application/json",
        },
      });
    } catch (networkError) {
      throw new Error(
        `Network error — make sure your device and the server are on the same Wi-Fi network.\nDetails: ${networkError.message}`,
      );
    }

    // ── Step 2: Validate HTTP status ──────────────────────────────────────
    if (!response.ok) {
      let serverMsg = `HTTP ${response.status}`;
      try {
        const errBody = await response.json();
        serverMsg = errBody.message || serverMsg;
      } catch (_) {
        /* ignore */
      }
      throw new Error(`Download failed: ${serverMsg}`);
    }

    // ── Step 3: Read response as text ─────────────────────────────────────
    const csvText = await response.text();

    if (!csvText || csvText.trim().length === 0) {
      throw new Error(
        "Server returned an empty file. No attendance data found for the selected period.",
      );
    }

    // Guard: detect if server accidentally returned a JSON error body
    if (csvText.trim().startsWith("{") || csvText.trim().startsWith("[")) {
      let msg = "Server returned an error instead of a CSV file.";
      try {
        const parsed = JSON.parse(csvText);
        msg = parsed.message || msg;
      } catch (_) {
        /* ignore */
      }
      throw new Error(msg);
    }

    // ── Step 4: Save file using stable expo-file-system API ──────────────
    const fileUri = FileSystem.documentDirectory + fileName;
    try {
      await FileSystem.writeAsStringAsync(fileUri, csvText, {
        encoding: FileSystem.EncodingType.UTF8,
      });
    } catch (writeError) {
      throw new Error(`Failed to save file: ${writeError.message}`);
    }

    // ── Step 5: Verify file was written ──────────────────────────────────
    const fileInfo = await FileSystem.getInfoAsync(fileUri);
    if (!fileInfo.exists || fileInfo.size === 0) {
      throw new Error(
        "File was saved but appears to be empty. Please try again.",
      );
    }

    // ── Step 6: Open native share dialog ─────────────────────────────────
    const canShare = await Sharing.isAvailableAsync();
    if (!canShare) {
      throw new Error("Sharing is not available on this device.");
    }

    await Sharing.shareAsync(fileUri, {
      mimeType: "text/csv",
      dialogTitle: `Share ${fileName}`,
      UTI: "public.comma-separated-values-text", // iOS only
    });

    return fileUri;
  },

  getStudentTimeline: async (month, year) => {
    const headers = await getHeaders();
    const response = await fetch(
      `${BASE_URL}/student/timeline?month=${month}&year=${year}`,
      { headers },
    );
    return response.json();
  },

  getStudentSessions: async (page = 1, limit = 15) => {
    const headers = await getHeaders();
    const response = await fetch(
      `${BASE_URL}/student/sessions?page=${page}&limit=${limit}`,
      { headers },
    );
    return response.json();
  },

  // Notifications
  // For 'individual' target: pass student_ids (array) for multi-select, or email (string) for legacy single
  sendNotification: async (classId, target, message, email = null, studentIds = null) => {
    const headers = await getHeaders();
    const body = { class_id: classId, target, message };
    if (email) body.email = email;
    if (studentIds && studentIds.length > 0) body.student_ids = studentIds;
    const response = await fetch(`${BASE_URL}/notifications/send`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });
    return response.json();
  },

  // Teacher Signup (role is always 'teacher' — students are created by teachers)
  signup: async (email, password, name, department) => {
    try {
      const response = await fetch(`${BASE_URL}/signup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email,
          password,
          name,
          role: "teacher",
          department,
        }),
      });
      const data = await response.json();
      if (data.status === "success") {
        return data;
      } else {
        throw new Error(data.message || "Signup failed");
      }
    } catch (error) {
      throw error;
    }
  },

  // Student Registration (Teacher only)
  registerStudent: async (
    name,
    roll_no,
    encoding,
    email,
    phone,
    department,
    batch,
    user_id,
  ) => {
    try {
      const headers = await getHeaders();
      const response = await fetch(`${BASE_URL}/students/register`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          name,
          roll_no,
          encoding,
          email,
          phone,
          department,
          batch,
          user_id,
        }),
      });
      const data = await response.json();
      if (data.status === "success") {
        return data;
      } else {
        throw new Error(data.message || "Registration failed");
      }
    } catch (error) {
      throw error;
    }
  },

  // Base URL for image upload
  BASE_URL,
};
