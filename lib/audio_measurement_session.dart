import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// record owns PCM/permission/interruption handling; only its missing mode API
/// goes native. Disable record.manageAudioSession before begin on iOS.
class AcousticAudioSession {
  static const _channel = MethodChannel('thud/acoustic_audio_session');
  static bool get supported => !kIsWeb && defaultTargetPlatform == TargetPlatform.iOS;
  static Future<void> begin() async {
    if (supported) await _channel.invokeMethod<void>('begin');
  }
  static Future<void> assertValid() async {
    if (supported) await _channel.invokeMethod<void>('assertValid');
  }
  static Future<void> end() async {
    if (supported) await _channel.invokeMethod<void>('end');
  }
}
