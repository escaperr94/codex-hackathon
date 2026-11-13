import 'package:material_ui/material_ui.dart';

abstract final class C {
  static const bg = Color(0xFFF7F7F2);
  static const ink = Color(0xFF20211F);
  static const muted = Color(0xFF646760);
  static const line = Color(0xFFE5E6DF);
  static const blue = Color(0xFF4265E8);
  static const pale = Color(0xFFF0F1EC);
  static const green = Color(0xFF207660);
  static const amber = Color(0xFF99651C);
  static const red = Color(0xFFAD4B4B);
  static const black = Color(0xFF151A20);
}

class Label extends StatelessWidget {
  const Label(this.text, {super.key, this.color = C.muted, this.fontSize = 12});
  final String text;
  final Color color;
  final double fontSize;
  @override
  Widget build(BuildContext context) {
    final value = text == text.toUpperCase() && text.length > 2
        ? '${text[0]}${text.substring(1).toLowerCase()}'
        : text;
    return Text(value,
        style: TextStyle(
            color: color,
            fontSize: fontSize < 12 ? 12 : fontSize,
            fontWeight: FontWeight.w500));
  }
}

class PageHeading extends StatelessWidget {
  const PageHeading(this.title, {super.key, this.subtitle, this.trailing});
  final String title;
  final String? subtitle;
  final Widget? trailing;
  @override
  Widget build(BuildContext context) =>
      Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Expanded(
            child:
                Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(title,
              style: const TextStyle(
                  color: C.ink,
                  fontSize: 34,
                  fontWeight: FontWeight.w600,
                  letterSpacing: -.6,
                  height: 1.1)),
          if (subtitle != null) ...[
            const SizedBox(height: 8),
            Text(subtitle!,
                style:
                    const TextStyle(color: C.muted, fontSize: 14, height: 1.45))
          ],
        ])),
        if (trailing != null) trailing!,
      ]);
}

class Panel extends StatelessWidget {
  const Panel(
      {required this.child,
      super.key,
      this.padding = const EdgeInsets.all(20),
      this.color = Colors.white});
  final Widget child;
  final EdgeInsets padding;
  final Color color;
  @override
  Widget build(BuildContext context) => Container(
      padding: padding,
      decoration: BoxDecoration(
          color: color,
          border: Border.all(color: C.line),
          borderRadius: BorderRadius.circular(24)),
      child: child);
}

class ActionButton extends StatefulWidget {
  const ActionButton(this.label,
      {required this.onPressed,
      super.key,
      this.icon = Icons.arrow_forward_rounded,
      this.secondary = false});
  final String label;
  final VoidCallback? onPressed;
  final IconData icon;
  final bool secondary;
  @override
  State<ActionButton> createState() => _ActionButtonState();
}

class _ActionButtonState extends State<ActionButton> {
  bool pressed = false;
  @override
  Widget build(BuildContext context) => Listener(
        onPointerDown: (_) {
          if (widget.onPressed != null) setState(() => pressed = true);
        },
        onPointerUp: (_) => setState(() => pressed = false),
        onPointerCancel: (_) => setState(() => pressed = false),
        child: AnimatedScale(
          scale: pressed ? .97 : 1,
          duration: Duration(
              milliseconds: MediaQuery.disableAnimationsOf(context) ? 0 : 130),
          curve: Curves.easeOutCubic,
          child: FilledButton.icon(
            onPressed: widget.onPressed,
            icon: Icon(widget.icon, size: 18),
            label: Text(widget.label),
            style: FilledButton.styleFrom(
                backgroundColor: widget.secondary ? C.pale : C.blue,
                foregroundColor: widget.secondary ? C.ink : Colors.white,
                minimumSize: const Size(48, 52),
                padding:
                    const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(26)),
                textStyle:
                    const TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
          ),
        ),
      );
}

class AppPage extends StatelessWidget {
  const AppPage(
      {required this.children,
      super.key,
      this.padding = const EdgeInsets.fromLTRB(24, 16, 24, 24)});
  final List<Widget> children;
  final EdgeInsets padding;
  @override
  Widget build(BuildContext context) => SingleChildScrollView(
      physics: const BouncingScrollPhysics(),
      padding: padding,
      child: Column(
          crossAxisAlignment: CrossAxisAlignment.start, children: children));
}

class SectionTitle extends StatelessWidget {
  const SectionTitle(this.title, {super.key, this.trailing});
  final String title;
  final Widget? trailing;
  @override
  Widget build(BuildContext context) => Row(children: [
        Expanded(
            child: Text(title,
                style: const TextStyle(
                    color: C.ink, fontSize: 16, fontWeight: FontWeight.w500))),
        if (trailing != null) trailing!
      ]);
}

class Metric extends StatelessWidget {
  const Metric(this.label, this.value, {super.key, this.accent = C.ink});
  final String label;
  final String value;
  final Color accent;
  @override
  Widget build(BuildContext context) => Expanded(
      child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 12),
          child:
              Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Label(label),
            const SizedBox(height: 8),
            Text(value,
                style: TextStyle(
                    color: accent,
                    fontSize: 24,
                    fontWeight: FontWeight.w500,
                    fontFeatures: const [FontFeature.tabularFigures()])),
          ])));
}

class StatusDot extends StatelessWidget {
  const StatusDot({super.key, this.color = C.green, this.size = 8});
  final Color color;
  final double size;
  @override
  Widget build(BuildContext context) => Container(
      width: size,
      height: size,
      decoration: BoxDecoration(color: color, shape: BoxShape.circle));
}

class MiniAction extends StatelessWidget {
  const MiniAction({required this.icon, required this.onPressed, super.key});
  final IconData icon;
  final VoidCallback onPressed;
  @override
  Widget build(BuildContext context) => IconButton(
      onPressed: onPressed,
      icon: Icon(icon, size: 20),
      color: C.muted,
      tooltip: 'Action');
}
