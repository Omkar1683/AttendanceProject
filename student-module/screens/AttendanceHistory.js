/**
 * student-module/screens/AttendanceHistory.js
 * ----------------------------------------------
 * Feature 4: Paginated attendance history with filter chips.
 */
import React, { useState, useEffect, useCallback } from 'react';
import {
  View, Text, ScrollView, StyleSheet, TouchableOpacity,
  RefreshControl, FlatList, ActivityIndicator,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Colors, FontSizes, Spacing, BorderRadius } from '../constants/theme';
import { ListSkeleton } from '../components/LoadingSkeleton';
import EmptyState from '../components/EmptyState';
import ErrorState from '../components/ErrorState';
import studentApi from '../services/studentApi';
import { formatDate, isToday, toDateString } from '../utils/dateUtils';

const FILTERS = [
  { key: 'all', label: 'All' },
  { key: 'today', label: 'Today' },
  { key: 'week', label: 'This Week' },
  { key: 'month', label: 'This Month' },
  { key: 'present', label: 'Present' },
  { key: 'absent', label: 'Absent' },
];

const AttendanceHistory = ({ navigateTo }) => {
  const [sessions, setSessions] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState('all');

  const loadData = useCallback(async (pg = 1, append = false) => {
    try {
      setError(null);
      const res = await studentApi.getSessions(pg, 30);
      if (res.status === 'success') {
        const data = res.data || {};
        if (append) {
          setSessions(prev => [...prev, ...(data.sessions || [])]);
        } else {
          setSessions(data.sessions || []);
        }
        setTotal(data.total || 0);
        setPage(pg);
      }
    } catch (err) {
      setError(err.message || 'Failed to load history');
    } finally {
      setLoading(false);
      setRefreshing(false);
      setLoadingMore(false);
    }
  }, []);

  useEffect(() => { loadData(); }, [loadData]);

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    loadData(1, false);
  }, [loadData]);

  const loadMore = () => {
    if (sessions.length < total && !loadingMore) {
      setLoadingMore(true);
      loadData(page + 1, true);
    }
  };

  // Filter sessions client-side
  const filteredSessions = sessions.filter(s => {
    if (filter === 'all') return true;
    if (filter === 'present') return s.status === 'Present';
    if (filter === 'absent') return s.status === 'Absent';
    if (filter === 'today') return isToday(s.date);
    if (filter === 'week') {
      const d = new Date(s.date);
      const now = new Date();
      const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
      return d >= weekAgo && d <= now;
    }
    if (filter === 'month') {
      const d = new Date(s.date);
      const now = new Date();
      return d.getMonth() === now.getMonth() && d.getFullYear() === now.getFullYear();
    }
    return true;
  });

  if (loading && !refreshing) return <ListSkeleton count={6} />;
  if (error) return <ErrorState message={error} onRetry={() => loadData(1)} />;

  return (
    <View style={histStyles.container}>
      {/* Filter Chips */}
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        style={histStyles.filterScroll}
        contentContainerStyle={histStyles.filterContent}
      >
        {FILTERS.map(f => (
          <TouchableOpacity
            key={f.key}
            style={[histStyles.chip, filter === f.key && histStyles.chipActive]}
            onPress={() => setFilter(f.key)}
            activeOpacity={0.7}
          >
            <Text style={[histStyles.chipText, filter === f.key && histStyles.chipTextActive]}>
              {f.label}
            </Text>
          </TouchableOpacity>
        ))}
      </ScrollView>

      {/* Count */}
      <Text style={histStyles.countText}>
        {filteredSessions.length} record{filteredSessions.length !== 1 ? 's' : ''}
        {filter !== 'all' ? ` (filtered)` : ` total`}
      </Text>

      {/* List */}
      <FlatList
        data={filteredSessions}
        keyExtractor={(item, index) => item.log_id || `${index}`}
        renderItem={({ item }) => (
          <View style={histStyles.card}>
            <View style={histStyles.cardLeft}>
              <View style={[histStyles.statusDot, {
                backgroundColor: item.status === 'Present' ? Colors.success : Colors.danger,
              }]} />
              <View>
                <Text style={histStyles.subject}>{item.subject}</Text>
                <Text style={histStyles.meta}>
                  {formatDate(item.date)} • {item.time}
                </Text>
                {item.faculty_name ? (
                  <Text style={histStyles.faculty}>👨‍🏫 {item.faculty_name}</Text>
                ) : null}
              </View>
            </View>
            <View style={histStyles.cardRight}>
              <View style={[histStyles.statusBadge, {
                backgroundColor: item.status === 'Present' ? Colors.successLight : Colors.dangerLight,
              }]}>
                <Text style={{
                  color: item.status === 'Present' ? Colors.success : Colors.danger,
                  fontSize: FontSizes.xs, fontWeight: 'bold',
                }}>
                  {item.status}
                </Text>
              </View>
              {item.marked_by === 'Manual' && (
                <Text style={histStyles.manualTag}>Manual</Text>
              )}
              {item.marked_by === 'AI' && item.confidence && (
                <Text style={histStyles.confidenceTag}>
                  AI {Math.round(item.confidence * 100)}%
                </Text>
              )}
            </View>
          </View>
        )}
        onEndReached={loadMore}
        onEndReachedThreshold={0.5}
        ListFooterComponent={loadingMore ? <ActivityIndicator style={{ padding: 16 }} color={Colors.primary} /> : null}
        ListEmptyComponent={
          <EmptyState
            icon="📋"
            title="No Records Found"
            subtitle={filter !== 'all' ? 'Try changing the filter' : 'No attendance records yet'}
          />
        }
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} colors={[Colors.primary]} />}
        showsVerticalScrollIndicator={false}
        contentContainerStyle={{ paddingHorizontal: Spacing.lg, paddingBottom: 24 }}
      />
    </View>
  );
};

const histStyles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  filterScroll: { maxHeight: 48 },
  filterContent: { paddingHorizontal: Spacing.lg, paddingVertical: Spacing.sm, gap: Spacing.sm },
  chip: {
    paddingHorizontal: Spacing.md, paddingVertical: Spacing.sm,
    borderRadius: BorderRadius.full, backgroundColor: Colors.surface,
    borderWidth: 1, borderColor: Colors.border,
  },
  chipActive: { backgroundColor: Colors.primary, borderColor: Colors.primary },
  chipText: { fontSize: FontSizes.sm, color: Colors.textSecondary, fontWeight: '500' },
  chipTextActive: { color: Colors.textWhite },
  countText: {
    fontSize: FontSizes.xs, color: Colors.textMuted,
    paddingHorizontal: Spacing.lg, paddingVertical: Spacing.sm,
  },
  card: {
    backgroundColor: Colors.surface, borderRadius: BorderRadius.lg,
    padding: Spacing.lg, marginBottom: Spacing.sm,
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    borderWidth: 1, borderColor: Colors.borderLight,
  },
  cardLeft: { flexDirection: 'row', alignItems: 'center', flex: 1 },
  statusDot: { width: 10, height: 10, borderRadius: 5, marginRight: Spacing.md },
  subject: { fontSize: FontSizes.md, fontWeight: '600', color: Colors.text },
  meta: { fontSize: FontSizes.xs, color: Colors.textMuted, marginTop: 2 },
  faculty: { fontSize: FontSizes.xs, color: Colors.textTertiary, marginTop: 2 },
  cardRight: { alignItems: 'flex-end' },
  statusBadge: {
    paddingHorizontal: Spacing.sm, paddingVertical: 2,
    borderRadius: BorderRadius.sm,
  },
  manualTag: { fontSize: 9, color: Colors.warning, marginTop: 3, fontWeight: '600' },
  confidenceTag: { fontSize: 9, color: Colors.textMuted, marginTop: 3 },
});

export default AttendanceHistory;
