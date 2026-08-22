/**
 * student-module/screens/StudentNotifications.js
 * -------------------------------------------------
 * Feature 5: Student notifications with read/unread state.
 */
import React, { useState, useEffect, useCallback } from 'react';
import {
  View, Text, FlatList, StyleSheet, TouchableOpacity,
  RefreshControl, ActivityIndicator,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Colors, FontSizes, Spacing, BorderRadius } from '../constants/theme';
import { ListSkeleton } from '../components/LoadingSkeleton';
import EmptyState from '../components/EmptyState';
import ErrorState from '../components/ErrorState';
import studentApi from '../services/studentApi';
import { formatDate } from '../utils/dateUtils';

const TARGET_ICONS = {
  all: { icon: 'megaphone-outline', color: Colors.primary },
  defaulters: { icon: 'warning-outline', color: Colors.danger },
  critical: { icon: 'alert-circle-outline', color: Colors.danger },
  individual: { icon: 'person-outline', color: Colors.accent },
};

const StudentNotifications = ({ onUnreadChange }) => {
  const [notifications, setNotifications] = useState([]);
  const [total, setTotal] = useState(0);
  const [unread, setUnread] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);

  const loadData = useCallback(async (pg = 1, append = false) => {
    try {
      setError(null);
      const res = await studentApi.getNotifications(pg, 30);
      if (res.status === 'success') {
        const data = res.data || {};
        if (append) {
          setNotifications(prev => [...prev, ...(data.notifications || [])]);
        } else {
          setNotifications(data.notifications || []);
        }
        setTotal(data.total || 0);
        setUnread(data.unread || 0);
        if (onUnreadChange) onUnreadChange(data.unread || 0);
        setPage(pg);
      }
    } catch (err) {
      setError(err.message || 'Failed to load notifications');
    } finally {
      setLoading(false);
      setRefreshing(false);
      setLoadingMore(false);
    }
  }, [onUnreadChange]);

  useEffect(() => { loadData(); }, [loadData]);

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    loadData(1, false);
  }, [loadData]);

  const loadMore = () => {
    if (notifications.length < total && !loadingMore) {
      setLoadingMore(true);
      loadData(page + 1, true);
    }
  };

  const handleMarkRead = async (notifId) => {
    try {
      await studentApi.markNotificationRead(notifId);
      setNotifications(prev =>
        prev.map(n => n.id === notifId ? { ...n, read: true } : n)
      );
      setUnread(prev => {
        const newVal = Math.max(0, prev - 1);
        if (onUnreadChange) onUnreadChange(newVal);
        return newVal;
      });
    } catch (err) {
      console.warn('Failed to mark notification read:', err);
    }
  };

  const getTimeAgo = (dateStr) => {
    if (!dateStr) return '';
    const now = new Date();
    const date = new Date(dateStr);
    const diff = now - date;
    const minutes = Math.floor(diff / 60000);
    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    if (days < 7) return `${days}d ago`;
    return formatDate(dateStr);
  };

  if (loading && !refreshing) return <ListSkeleton count={5} />;
  if (error) return <ErrorState message={error} onRetry={() => loadData(1)} />;

  return (
    <View style={notifStyles.container}>
      {/* Header Stats */}
      <View style={notifStyles.headerStats}>
        <View style={notifStyles.statItem}>
          <Text style={notifStyles.statValue}>{total}</Text>
          <Text style={notifStyles.statLabel}>Total</Text>
        </View>
        <View style={notifStyles.statItem}>
          <Text style={[notifStyles.statValue, { color: Colors.danger }]}>{unread}</Text>
          <Text style={notifStyles.statLabel}>Unread</Text>
        </View>
      </View>

      <FlatList
        data={notifications}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => {
          const targetInfo = TARGET_ICONS[item.target] || TARGET_ICONS.all;
          return (
            <TouchableOpacity
              style={[notifStyles.card, !item.read && notifStyles.cardUnread]}
              onPress={() => !item.read && handleMarkRead(item.id)}
              activeOpacity={item.read ? 1 : 0.7}
            >
              <View style={[notifStyles.iconCircle, { backgroundColor: targetInfo.color + '15' }]}>
                <Ionicons name={targetInfo.icon} size={20} color={targetInfo.color} />
              </View>
              <View style={notifStyles.cardContent}>
                <View style={notifStyles.cardHeader}>
                  <Text style={notifStyles.className}>{item.class_name}</Text>
                  <Text style={notifStyles.timeAgo}>{getTimeAgo(item.created_at)}</Text>
                </View>
                <Text style={notifStyles.message} numberOfLines={3}>{item.message}</Text>
                <View style={notifStyles.cardFooter}>
                  <Text style={notifStyles.teacher}>From: {item.teacher_name}</Text>
                  {!item.read && (
                    <View style={notifStyles.unreadDot} />
                  )}
                </View>
              </View>
            </TouchableOpacity>
          );
        }}
        onEndReached={loadMore}
        onEndReachedThreshold={0.5}
        ListFooterComponent={
          loadingMore ? <ActivityIndicator style={{ padding: 16 }} color={Colors.primary} /> : null
        }
        ListEmptyComponent={
          <EmptyState
            icon="🔔"
            title="No Notifications"
            subtitle="You're all caught up! Notifications from your teachers will appear here."
          />
        }
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} colors={[Colors.primary]} />}
        showsVerticalScrollIndicator={false}
        contentContainerStyle={{ paddingHorizontal: Spacing.lg, paddingBottom: 24 }}
      />
    </View>
  );
};

const notifStyles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  headerStats: {
    flexDirection: 'row', justifyContent: 'space-around',
    paddingVertical: Spacing.md, backgroundColor: Colors.surface,
    marginHorizontal: Spacing.lg, marginTop: Spacing.md,
    borderRadius: BorderRadius.lg, borderWidth: 1, borderColor: Colors.border,
    marginBottom: Spacing.sm,
  },
  statItem: { alignItems: 'center' },
  statValue: { fontSize: FontSizes.xxl, fontWeight: 'bold', color: Colors.text },
  statLabel: { fontSize: FontSizes.xs, color: Colors.textMuted },
  card: {
    backgroundColor: Colors.surface, borderRadius: BorderRadius.lg,
    padding: Spacing.lg, marginBottom: Spacing.sm,
    flexDirection: 'row', borderWidth: 1, borderColor: Colors.borderLight,
  },
  cardUnread: {
    borderLeftWidth: 3, borderLeftColor: Colors.primary,
    backgroundColor: Colors.primaryLight,
  },
  iconCircle: {
    width: 40, height: 40, borderRadius: 20,
    alignItems: 'center', justifyContent: 'center', marginRight: Spacing.md,
  },
  cardContent: { flex: 1 },
  cardHeader: {
    flexDirection: 'row', justifyContent: 'space-between',
    marginBottom: 4,
  },
  className: { fontSize: FontSizes.sm, fontWeight: 'bold', color: Colors.primary },
  timeAgo: { fontSize: FontSizes.xs, color: Colors.textMuted },
  message: { fontSize: FontSizes.md, color: Colors.text, lineHeight: 20, marginBottom: Spacing.xs },
  cardFooter: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  teacher: { fontSize: FontSizes.xs, color: Colors.textTertiary },
  unreadDot: {
    width: 8, height: 8, borderRadius: 4, backgroundColor: Colors.primary,
  },
});

export default StudentNotifications;
