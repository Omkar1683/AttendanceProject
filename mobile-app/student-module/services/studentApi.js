/**
 * student-module/services/studentApi.js
 * --------------------------------------
 * Student-specific API client.
 * Reuses the BASE_URL and auth token from the existing mobile-app/utils/api.js.
 */
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as FileSystem from 'expo-file-system';
import * as Sharing from 'expo-sharing';

// Same backend URL as the teacher module — must stay in sync
const BASE_URL = 'https://attendai-j5a4.onrender.com';
const WS_URL = 'wss://attendai-j5a4.onrender.com';

const getHeaders = async () => {
  const token = await AsyncStorage.getItem('userToken');
  const headers = { 'Content-Type': 'application/json' };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  return headers;
};

const studentApi = {
  BASE_URL,
  WS_URL,

  // ── Profile ──────────────────────────────────────────────────────────────
  getProfile: async () => {
    const headers = await getHeaders();
    const res = await fetch(`${BASE_URL}/student/profile`, { headers });
    return res.json();
  },

  updateProfile: async (data) => {
    const headers = await getHeaders();
    const res = await fetch(`${BASE_URL}/student/profile`, {
      method: 'PUT',
      headers,
      body: JSON.stringify(data),
    });
    return res.json();
  },

  changePassword: async (oldPassword, newPassword) => {
    const headers = await getHeaders();
    const res = await fetch(`${BASE_URL}/student/change-password`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ old_password: oldPassword, new_password: newPassword }),
    });
    return res.json();
  },

  // ── Attendance Report ────────────────────────────────────────────────────
  getReport: async () => {
    const headers = await getHeaders();
    const res = await fetch(`${BASE_URL}/student/report`, { headers });
    return res.json();
  },

  // ── Timeline (Calendar) ──────────────────────────────────────────────────
  getTimeline: async (month, year) => {
    const headers = await getHeaders();
    const res = await fetch(
      `${BASE_URL}/student/timeline?month=${month}&year=${year}`,
      { headers }
    );
    return res.json();
  },

  // ── Session History ──────────────────────────────────────────────────────
  getSessions: async (page = 1, limit = 20) => {
    const headers = await getHeaders();
    const res = await fetch(
      `${BASE_URL}/student/sessions?page=${page}&limit=${limit}`,
      { headers }
    );
    return res.json();
  },

  // ── Notifications ────────────────────────────────────────────────────────
  getNotifications: async (page = 1, limit = 30) => {
    const headers = await getHeaders();
    const res = await fetch(
      `${BASE_URL}/student/notifications?page=${page}&limit=${limit}`,
      { headers }
    );
    return res.json();
  },

  markNotificationRead: async (notificationId) => {
    const headers = await getHeaders();
    const res = await fetch(
      `${BASE_URL}/student/notifications/${notificationId}/read`,
      { method: 'POST', headers }
    );
    return res.json();
  },

  // ── Analytics ────────────────────────────────────────────────────────────
  getAnalytics: async () => {
    const headers = await getHeaders();
    const res = await fetch(`${BASE_URL}/student/analytics`, { headers });
    return res.json();
  },

  // ── Classes (enrolled) ───────────────────────────────────────────────────
  getEnrolledClasses: async () => {
    const headers = await getHeaders();
    const res = await fetch(`${BASE_URL}/student/classes`, { headers });
    return res.json();
  },

  // ── Export ───────────────────────────────────────────────────────────────
  exportCSV: async (studentName = 'student') => {
    const token = await AsyncStorage.getItem('userToken');
    if (!token) throw new Error('Not authenticated');

    const res = await fetch(`${BASE_URL}/student/export/csv`, {
      headers: {
        'Authorization': `Bearer ${token}`,
        'Accept': 'text/csv',
      },
    });

    if (!res.ok) {
      let msg = `HTTP ${res.status}`;
      try { const e = await res.json(); msg = e.message || msg; } catch {}
      throw new Error(msg);
    }

    const csvText = await res.text();
    if (!csvText || csvText.trim().length === 0) {
      throw new Error('No attendance data available to export.');
    }

    const safeName = (studentName || 'student').replace(/\s+/g, '_');
    const fileName = `attendance_${safeName}_${new Date().toISOString().split('T')[0]}.csv`;
    const fileUri = FileSystem.documentDirectory + fileName;

    await FileSystem.writeAsStringAsync(fileUri, csvText, {
      encoding: FileSystem.EncodingType.UTF8,
    });

    const canShare = await Sharing.isAvailableAsync();
    if (canShare) {
      await Sharing.shareAsync(fileUri, {
        mimeType: 'text/csv',
        dialogTitle: `Share ${fileName}`,
        UTI: 'public.comma-separated-values-text',
      });
    }

    return fileUri;
  },

  exportPDF: async (studentName = 'student') => {
    const token = await AsyncStorage.getItem('userToken');
    if (!token) throw new Error('Not authenticated');

    const res = await fetch(`${BASE_URL}/student/export/pdf`, {
      headers: {
        'Authorization': `Bearer ${token}`,
        'Accept': 'application/pdf',
      },
    });

    if (!res.ok) {
      let msg = `HTTP ${res.status}`;
      try { const e = await res.json(); msg = e.message || msg; } catch {}
      throw new Error(msg);
    }

    // Read as base64 since PDF is binary
    const blob = await res.blob();
    const reader = new FileReader();
    const base64 = await new Promise((resolve, reject) => {
      reader.onloadend = () => {
        const base64data = reader.result.split(',')[1];
        resolve(base64data);
      };
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });

    const safeName = (studentName || 'student').replace(/\s+/g, '_');
    const fileName = `attendance_${safeName}_${new Date().toISOString().split('T')[0]}.pdf`;
    const fileUri = FileSystem.documentDirectory + fileName;

    await FileSystem.writeAsStringAsync(fileUri, base64, {
      encoding: FileSystem.EncodingType.Base64,
    });

    const canShare = await Sharing.isAvailableAsync();
    if (canShare) {
      await Sharing.shareAsync(fileUri, {
        mimeType: 'application/pdf',
        dialogTitle: `Share ${fileName}`,
        UTI: 'com.adobe.pdf',
      });
    }

    return fileUri;
  },

  // ── WebSocket ────────────────────────────────────────────────────────────
  connectStudentSocket: (studentId, onAttendanceUpdate) => {
    const wsUrl = `${WS_URL}/socket.io/?EIO=4&transport=websocket`;
    let ws;
    let closed = false;

    const connect = () => {
      if (closed) return;
      ws = new WebSocket(wsUrl);

      const sendJoin = () => {
        try {
          ws.send(`42["join_student",{"student_id":"${studentId}"}]`);
        } catch {}
      };

      ws.onopen = () => {
        console.log('[StudentSocket] Connected');
        sendJoin();
      };

      ws.onmessage = (event) => {
        const msg = event.data;
        if (typeof msg === 'string' && (msg.startsWith('0') || msg.startsWith('40'))) {
          sendJoin();
          return;
        }
        if (msg === '2') { ws.send('3'); return; }
        if (typeof msg === 'string' && msg.startsWith('42')) {
          try {
            const [eventName, payload] = JSON.parse(msg.slice(2));
            if (eventName === 'student_attendance_update' && onAttendanceUpdate) {
              onAttendanceUpdate(payload);
            }
          } catch {}
        }
      };

      ws.onclose = () => {
        if (!closed) {
          console.log('[StudentSocket] Disconnected — reconnecting in 3s');
          setTimeout(connect, 3000);
        }
      };

      ws.onerror = (err) => {
        console.warn('[StudentSocket] Error:', err.message);
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
};

export default studentApi;
