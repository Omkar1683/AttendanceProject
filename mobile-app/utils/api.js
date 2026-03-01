
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Alert } from 'react-native';

// Backend server URL - ensure your mobile device is on the same network
const BASE_URL = 'http://192.168.29.56:5000';

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

    // Notifications
    sendNotification: async (classId, target, message) => {
        const headers = await getHeaders();
        const response = await fetch(`${BASE_URL}/notifications/send`, {
            method: 'POST',
            headers,
            body: JSON.stringify({ class_id: classId, target, message }),
        });
        return response.json();
    },

    // User Registration
    signup: async (email, password, name, role, department, roll_no = null) => {
        try {
            const response = await fetch(`${BASE_URL}/signup`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password, name, role, department, roll_no }),
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
