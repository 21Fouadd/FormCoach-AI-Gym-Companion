import 'package:flutter/material.dart';
import '../models/social.dart';
import '../theme/app_theme.dart';
import '../widgets/common.dart';
import 'community_screen.dart' show Avatar, RoleBadge;

class GroupDetailScreen extends StatefulWidget {
  final Group group;
  const GroupDetailScreen({super.key, required this.group});

  @override
  State<GroupDetailScreen> createState() => _GroupDetailScreenState();
}

class _GroupDetailScreenState extends State<GroupDetailScreen> {
  late List<GroupMember> members = [...widget.group.members];

  bool get canManage => widget.group.joined && widget.group.myRole.canManage;
  GroupRole get myRole => widget.group.myRole;

  /// Whether you have any management action on a member with [target] role.
  bool _hasActionsOn(GroupRole target) =>
      widget.group.joined && myRole.hasActionsOn(target);

  void _setRole(int i, GroupRole role) {
    setState(() => members[i] = GroupMember(
        members[i].person, role, members[i].weeklyReps));
  }

  void _memberMenu(int i) {
    final m = members[i];
    if (!_hasActionsOn(m.role)) return;
    final canChangeRole = myRole.canChangeRoleOf(m.role);
    showModalBottomSheet(
      context: context,
      backgroundColor: AppColors.surface,
      builder: (_) => SafeArea(
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          ListTile(
              title: Text(m.person.name,
                  style: const TextStyle(fontWeight: FontWeight.w700))),
          if (canChangeRole && m.role == GroupRole.member)
            ListTile(
              leading: const Icon(Icons.shield_outlined),
              title: const Text('Make admin'),
              onTap: () {
                _setRole(i, GroupRole.admin);
                Navigator.pop(context);
              },
            ),
          if (canChangeRole && m.role == GroupRole.admin)
            ListTile(
              leading: const Icon(Icons.remove_moderator_outlined),
              title: const Text('Remove admin'),
              onTap: () {
                _setRole(i, GroupRole.member);
                Navigator.pop(context);
              },
            ),
          if (myRole.canRemove(m.role))
            ListTile(
              leading: const Icon(Icons.person_remove_outlined,
                  color: AppColors.bad),
              title: const Text('Remove from group',
                  style: TextStyle(color: AppColors.bad)),
              onTap: () {
                setState(() => members.removeAt(i));
                Navigator.pop(context);
              },
            ),
        ]),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final g = widget.group;
    final ranked = [...members]..sort((a, b) => b.weeklyReps.compareTo(a.weeklyReps));

    return Scaffold(
      appBar: AppBar(
        title: Text(g.name, style: const TextStyle(fontWeight: FontWeight.w800)),
        actions: [
          if (canManage)
            IconButton(
                onPressed: () {}, icon: const Icon(Icons.settings_outlined)),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(AppSpace.md),
        children: [
          Panel(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(children: [
                  Icon(g.isPrivate ? Icons.lock_outline : Icons.public,
                      size: 16, color: AppColors.textDim),
                  const SizedBox(width: 6),
                  Expanded(
                    child: Text(
                        '${g.isPrivate ? 'Private group' : 'Public community'}  -  ${g.memberCount} members',
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(
                            color: AppColors.textDim, fontSize: 13)),
                  ),
                ]),
                const SizedBox(height: 10),
                Text(g.description),
                const SizedBox(height: 16),
                if (!g.joined)
                  FilledButton(onPressed: () {}, child: const Text('Join'))
                else if (canManage)
                  Row(children: [
                    Expanded(
                      child: FilledButton.icon(
                        onPressed: () {},
                        icon: const Icon(Icons.person_add_alt),
                        label: const Text('Invite'),
                      ),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: OutlinedButton.icon(
                        onPressed: () {},
                        icon: const Icon(Icons.campaign_outlined),
                        label: const Text('Announce'),
                      ),
                    ),
                  ])
                else
                  OutlinedButton(
                      onPressed: () {}, child: const Text('Leave group')),
              ],
            ),
          ),
          if (members.isNotEmpty) ...[
            const SectionTitle('This week leaderboard'),
            for (var i = 0; i < ranked.length && i < 3; i++)
              Padding(
                padding: const EdgeInsets.only(bottom: 10),
                child: Panel(
                  child: Row(children: [
                    SizedBox(
                      width: 28,
                      child: Text('${i + 1}',
                          style: TextStyle(
                              fontSize: 20,
                              fontWeight: FontWeight.w900,
                              color: i == 0 ? AppColors.accent : AppColors.textDim)),
                    ),
                    Avatar(ranked[i].person, size: 40),
                    const SizedBox(width: 12),
                    Expanded(
                        child: Text(ranked[i].person.name,
                            style: const TextStyle(fontWeight: FontWeight.w700))),
                    Text('${ranked[i].weeklyReps} reps',
                        style: const TextStyle(fontWeight: FontWeight.w700)),
                  ]),
                ),
              ),
            SectionTitle('Members (${members.length})',
                action: canManage ? 'Manage' : null),
            for (var i = 0; i < members.length; i++)
              Padding(
                padding: const EdgeInsets.only(bottom: 10),
                child: Panel(
                  onTap: _hasActionsOn(members[i].role)
                      ? () => _memberMenu(i)
                      : null,
                  child: Row(children: [
                    Avatar(members[i].person, size: 40),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Row(children: [
                        Flexible(
                          child: Text(members[i].person.name,
                              overflow: TextOverflow.ellipsis,
                              style: const TextStyle(fontWeight: FontWeight.w700)),
                        ),
                        RoleBadge(members[i].role),
                      ]),
                    ),
                    if (_hasActionsOn(members[i].role))
                      const Icon(Icons.more_horiz, color: AppColors.textDim),
                  ]),
                ),
              ),
          ],
        ],
      ),
    );
  }
}
