if (norm(gradA-gradASave) > 10^-12) fprintf('GradA: %.4e\n',norm(gradA-gradASave)); end;
if (norm(gradB-gradBSave) > 10^-12) fprintf('GradB: %.4e\n',norm(gradB-gradBSave)); end;
if (abs(TermA-TermASave) > 10^-12) fprintf('TermA: %.4e\n',TermA-TermASave); end;
if (abs(TermB - TermBSave) > 10^-12) fprintf('TermB: %.4e\n',TermB-TermBSave); end;
