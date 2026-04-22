import AsyncStorage from '@react-native-async-storage/async-storage';
import * as FileSystem from 'expo-file-system';
import { File, Directory, Paths } from 'expo-file-system/next';
import * as Sharing from 'expo-sharing';

// Backend server URL - ensure your mobile device is on the same network
const BASE_URL = 'http://192.168.167.145:5000';


const getHeaders = async () => {
    const token = await AsyncStorage.getItem('userToken');
    const headers = {
        'Content-Type': 'application/json',
    };
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }
    return headers;
};

export const api = {
    // Auth
    login: async (email, password) => {
        try {
            const response = await fetch(`${BASE_URL}/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password }),
            });
            const data = await response.json();
            if (data.status === 'success') {
                await AsyncStorage.setItem('userToken', data.token);
                await AsyncStorage.setItem('userInfo', JSON.stringify(data.user));
                return data;
            } else {
                throw new Error(data.message || 'Login failed');
            }
        } catch (error) {
            throw error;
        }
    },

    logout: async () => {
        await AsyncStorage.removeItem('userToken');
        await AsyncStorage.removeItem('userInfo');
    },

    getUserInfo: async () => {
        const info = await AsyncStorage.getItem('userInfo');
        return info ? JSON.parse(info) : null;
    },

    // Data Fetching
    getClasses: async (teacherId) => {
        const headers = await getHeaders();
        const response = await fetch(`${BASE_URL}/classes?teacher_id=${teacherId}`, { headers });
        return response.json();
    },

    // ── Create Subject/Class ──────────────────────────────────────────────────────
    createClass: async ({ name, code, student_ids = [], batch, department, schedule }) => {
        const headers = await getHeaders();
        // total_students is derived from selected students count (or 1 minimum)
        const total_students = Math.max(student_ids.length, 1);
        const response = await fetch(`${BASE_URL}/classes/create`, {
            method: 'POST',
            headers,
            body: JSON.stringify({ name, code, total_students, student_ids, batch, department, schedule }),
        });
        return response.json();
    },

    // ── Get all students (for class assignment picker) ────────────────────────────
    getStudents: async (batch = '') => {
        const headers = await getHeaders();
        const query = batch ? `?batch=${encodeURIComponent(batch)}` : '';
        const response = await fetch(`${BASE_URL}/students${query}`, { headers });
        return response.json();
    },


    getTodayAnalytics: async (classId) => {
        const headers = await getHeaders();
        const response = await fetch(`${BASE_URL}/analytics/today?class_id=${classId}`, { headers });
        return response.json();
    },

    getDefaulters: async (classId, threshold = 75) => {
        const headers = await getHeaders();
        const response = await fetch(`${BASE_URL}/analytics/defaulters?class_id=${classId}&threshold=${threshold}`, { headers });
        return response.json();
    },

    // Session Management
    createSession: async (classId, location) => {
        const headers = await getHeaders();
        const response = await fetch(`${BASE_URL}/sessions/create`, {
            method: 'POST',
            headers,
            body: JSON.stringify({ class_id: classId, location }),
        });
        return response.json();
    },

    stopSession: async (sessionId) => {
        const headers = await getHeaders();
        const response = await fetch(`${BASE_URL}/sessions/stop`, {
            method: 'POST',
            headers,
            body: JSON.stringify({ session_id: sessionId }),
        });
        return response.json();
    },

    // Reports
    getClassReport: async (classId, month, year) => {
        const headers = await getHeaders();
        const response = await fetch(`${BASE_URL}/reports/class?class_id=${classId}&month=${month}&year=${year}`, { headers });
        return response.json();
    },

    getStudentReport: async (studentId) => {
        const headers = await getHeaders();
        const response = await fetch(`${BASE_URL}/reports/student/${studentId}`, { headers });
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
            method: 'PUT',
            headers,
            body: JSON.stringify(data),
        });
        return response.json();
    },

    changePassword: async (old_password, new_password) => {
        const headers = await getHeaders();
        const response = await fetch(`${BASE_URL}/student/change-password`, {
            method: 'POST',
            headers,
            body: JSON.stringify({ old_password, new_password }),
        });
        return response.json();
    },

    // ── NEW: Manual Attendance Edit ─────────────────────────────────────────────────
    manualAttendance: async (studentId, sessionId, status) => {
        const headers = await getHeaders();
        const response = await fetch(`${BASE_URL}/attendance/manual`, {
            method: 'POST',
            headers,
            body: JSON.stringify({
                student_id: studentId,
                session_id: sessionId,
                status,          // 'Present' or 'Absent'
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
            { headers }
        );
        return response.json();
    },

    // ── CSV Download & Share ──────────────────────────────────────────────────────────
    // SDK 54+: uses new File class from expo-file-system/next.
    // Old APIs (downloadAsync, writeAsStringAsync) are fully deprecated in SDK 54.
    downloadAndShareCSV: async (classId, month, year, className) => {
        const token = await AsyncStorage.getItem('userToken');

        if (!token) {
            throw new Error('Authentication token missing. Please log in again.');
        }
        if (!classId) {
            throw new Error('Class ID is required to export the report.');
        }

        const url = `${BASE_URL}/reports/export-csv?class_id=${classId}&month=${month}&year=${year}`;
        const safeClassName = (className || 'report').replace(/\s+/g, '_');
        const fileName = `attendance_${safeClassName}_${month}_${year}.csv`;

        // ── Step 1: Fetch CSV from backend ────────────────────────────────────
        let response;
        try {
            response = await fetch(url, {
                method: 'GET',
                headers: {
                    Authorization: `Bearer ${token}`,
                    Accept: 'text/csv, application/json',
                },
            });
        } catch (networkError) {
            throw new Error(
                `Network error — make sure your device and the server are on the same Wi-Fi network.\nDetails: ${networkError.message}`
            );
        }

        // ── Step 2: Validate HTTP status ──────────────────────────────────────
        if (!response.ok) {
            let serverMsg = `HTTP ${response.status}`;
            try {
                const errBody = await response.json();
                serverMsg = errBody.message || serverMsg;
            } catch (_) { /* ignore */ }
            throw new Error(`Download failed: ${serverMsg}`);
        }

        // ── Step 3: Read response as text ─────────────────────────────────────
        const csvText = await response.text();

        if (!csvText || csvText.trim().length === 0) {
            throw new Error('Server returned an empty file. No attendance data found for the selected period.');
        }

        // Guard: detect if server accidentally returned a JSON error body
        if (csvText.trim().startsWith('{') || csvText.trim().startsWith('[')) {
            let msg = 'Server returned an error instead of a CSV file.';
            try {
                const parsed = JSON.parse(csvText);
                msg = parsed.message || msg;
            } catch (_) { /* ignore */ }
            throw new Error(msg);
        }

        // ── Step 4: Save using NEW File API (expo-file-system/next) ───────────
        // Paths.document is the app's document directory (persistent, shareable)
        const csvFile = new File(Paths.document, fileName);
        try {
            csvFile.write(csvText);
        } catch (writeError) {
            throw new Error(`Failed to save file: ${writeError.message}`);
        }

        // ── Step 5: Verify file was written ──────────────────────────────────
        if (!csvFile.exists || csvFile.size === 0) {
            throw new Error('File was saved but appears to be empty. Please try again.');
        }

        // ── Step 6: Open native share dialog ─────────────────────────────────
        const canShare = await Sharing.isAvailableAsync();
        if (!canShare) {
            throw new Error('Sharing is not available on this device.');
        }

        await Sharing.shareAsync(csvFile.uri, {
            mimeType: 'text/csv',
            dialogTitle: `Share ${fileName}`,
            UTI: 'public.comma-separated-values-text', // iOS only
        });

        return csvFile.uri;
    },

    getStudentTimeline: async (month, year) => {
        const headers = await getHeaders();
        const response = await fetch(
            `${BASE_URL}/student/timeline?month=${month}&year=${year}`,
            { headers }
        );
        return response.json();
    },

    getStudentSessions: async (page = 1, limit = 15) => {
        const headers = await getHeaders();
        const response = await fetch(
            `${BASE_URL}/student/sessions?page=${page}&limit=${limit}`,
            { headers }
        );
        return response.json();
    },

    // Notifications
    // NOTE: for 'individual' target, pass the student's email in the `email` param
    sendNotification: async (classId, target, message, email = null) => {
        const headers = await getHeaders();
        const response = await fetch(`${BASE_URL}/notifications/send`, {
            method: 'POST',
            headers,
            body: JSON.stringify({ class_id: classId, target, message, email }),
        });
        return response.json();
    },

    // Teacher Signup (role is always 'teacher' — students are created by teachers)
    signup: async (email, password, name, department) => {
        try {
            const response = await fetch(`${BASE_URL}/signup`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password, name, role: 'teacher', department }),
            });
            const data = await response.json();
            if (data.status === 'success') {
                return data;
            } else {
                throw new Error(data.message || 'Signup failed');
            }
        } catch (error) {
            throw error;
        }
    },

    // Student Registration (Teacher only)
    registerStudent: async (name, roll_no, encoding, email, phone, department, batch, user_id) => {
        try {
            const headers = await getHeaders();
            const response = await fetch(`${BASE_URL}/students/register`, {
                method: 'POST',
                headers,
                body: JSON.stringify({
                    name,
                    roll_no,
                    encoding,
                    email,
                    phone,
                    department,
                    batch,
                    user_id
                }),
            });
            const data = await response.json();
            if (data.status === 'success') {
                return data;
            } else {
                throw new Error(data.message || 'Registration failed');
            }
        } catch (error) {
            throw error;
        }
    },

    // Base URL for image upload
    BASE_URL
};
